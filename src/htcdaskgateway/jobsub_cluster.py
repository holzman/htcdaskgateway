import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

from .cluster import HTCGatewayCluster

logger = logging.getLogger("htcdaskgateway.JobsubGatewayCluster")


class JobsubGatewayCluster(HTCGatewayCluster):
    """
    A GatewayCluster subclass for sites that use jobsub_lite
    (jobsub_submit / jobsub_rm) for HTCondor job submission and do not have a
    shared NFS filesystem between the submit node and execute nodes.

    Unlike HTCGatewayCluster (which writes job sandboxes to an NFS path under
    /uscmst1b_scratch/), this class creates a local temporary directory per
    scale_batch_workers() call and uses HTCondor native file transfer to ship
    credentials into the job sandbox.

    X.509 proxy transfer is omitted — jobsub_submit uses SciTokens /
    bearer-token authentication for job submission.

    Parameters
    ----------
    jobsub_group : str, optional
        The ``-G`` group argument required by every ``jobsub_submit`` /
        ``jobsub_rm`` call.  Defaults to the value of the ``$GROUP``
        environment variable, falling back to ``$EXPERIMENT``.  A
        ``ValueError`` is raised at construction time if none of these is set.
    apptainer_bin : str, optional
        Full path to the Apptainer / Singularity binary.  Differs between
        sites and experiments; defaults to the OSG-supplied binary on CVMFS.
    api_url : str, optional
        Value of ``DASK_GATEWAY_API_URL`` set inside the worker container.
        Defaults to the EAF endpoint.
    scheduler_proxy_ip : str, optional
        IP address of the Dask Gateway scheduler proxy.  Exposed as a proper
        constructor parameter here because the parent class reads it via a
        broken ``kwargs.pop('', ...)`` call (empty-string key) that never
        matches a real kwarg, leaving it always hardcoded.  Defaults to
        ``131.225.218.222``; override for other deployments.
    node_match_expr : str, optional
        Raw HTCondor classad expression injected via ``--lines`` to
        ``jobsub_submit``.  Must be a valid classad expression (validated by
        ``classad.parseOne()`` inside jobsub_lite).  Pass an empty string
        ``''`` to disable node matching entirely.  Default is
        ``'rank = (isDaskNode == True)'``.
    **kwargs
        All remaining keyword arguments are forwarded to
        ``HTCGatewayCluster.__init__``.
    """

    def __init__(
        self,
        jobsub_group=None,
        apptainer_bin="/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer",
        api_url="https://dask-gateway-api.fnal.gov/api",
        scheduler_proxy_ip="131.225.218.222",
        node_match_expr="rank = (isDaskNode == True)",
        **kwargs,
    ):
        # Resolve jobsub group: constructor arg → $GROUP → $EXPERIMENT → error
        if jobsub_group is None:
            jobsub_group = os.environ.get("GROUP") or os.environ.get("EXPERIMENT")
        if not jobsub_group:
            raise ValueError(
                "jobsub_group must be provided either as a constructor argument "
                "or via the $GROUP / $EXPERIMENT environment variable."
            )
        self.jobsub_group = jobsub_group
        self.apptainer_bin = apptainer_bin
        self.api_url = api_url
        # Override the parent's (broken) scheduler_proxy_ip before super().__init__
        # is called, so that if the parent ever reads self.scheduler_proxy_ip it sees
        # the correct value.  We also store it here explicitly.
        self.scheduler_proxy_ip = scheduler_proxy_ip

        # One entry appended per scale_batch_workers() call; kept alive for the
        # full cluster lifetime so HTCondor can re-transfer on eviction/restart.
        self._tmpdirs = []

        super().__init__(
            node_match_expr=node_match_expr,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def scale_batch_workers(self, n):
        """Submit ``n`` HTCondor workers via ``jobsub_submit``."""
        security = self.security
        cluster_name = self.name

        # Build image path (same convention as parent)
        image_name = (
            "/cvmfs/unpacked.cern.ch/"
            + self.image_registry
            + "/"
            + self.apptainer_image
        )

        # Resolve memory / cores (mirror parent's fallback to gateway defaults)
        if self.worker_memory:
            worker_mem = f"{self.worker_memory}GB"
            print("Using Specified worker_memory:", worker_mem)
        else:
            options = self.gateway.cluster_options()
            worker_mem = f"{options.worker_memory}GB"
            print("Using Default worker_memory:", worker_mem)

        if self.worker_cores:
            num_cores = str(self.worker_cores)
            print("Using Specified worker_cores:", num_cores, "cores")
        else:
            options = self.gateway.cluster_options()
            num_cores = str(options.worker_cores)
            print("Using Default worker_cores:", num_cores, "cores")

        # ------------------------------------------------------------------
        # Step 1 — allocate a local scratch directory for this scale call
        # ------------------------------------------------------------------
        tmpdir = tempfile.mkdtemp(prefix=f"htcdask-{cluster_name}-")
        os.chmod(tmpdir, 0o700)
        self._tmpdirs.append(tmpdir)

        credentials_dir = f"{tmpdir}/dask-credentials"
        worker_space_dir = f"{tmpdir}/dask-worker-space"
        condor_logdir = f"{tmpdir}/condor"

        os.makedirs(credentials_dir, mode=0o700, exist_ok=True)
        os.makedirs(worker_space_dir, mode=0o700, exist_ok=True)
        os.makedirs(condor_logdir, mode=0o700, exist_ok=True)

        # ------------------------------------------------------------------
        # Step 2 — write credentials
        # ------------------------------------------------------------------
        with open(f"{credentials_dir}/dask.crt", "w") as f:
            f.write(security.tls_cert)
        with open(f"{credentials_dir}/dask.pem", "w") as f:
            f.write(security.tls_key)
        with open(f"{credentials_dir}/api-token", "w") as f:
            f.write(os.environ["JUPYTERHUB_API_TOKEN"])

        # ------------------------------------------------------------------
        # Step 3 — write start.sh
        # No x509 proxy copy (SciTokens / bearer-token auth used instead).
        # ------------------------------------------------------------------
        start_sh = f"""#!/bin/bash
export APPTAINERENV_DASK_GATEWAY_WORKER_NAME=$2
export APPTAINERENV_DASK_GATEWAY_API_URL="{self.api_url}"
export APPTAINERENV_DASK_GATEWAY_CLUSTER_NAME=$1
export APPTAINERENV_DASK_GATEWAY_API_TOKEN=/etc/dask-credentials/api-token

worker_space_dir=${{PWD}}/dask-worker-space/$2
mkdir $worker_space_dir

{self.apptainer_bin} exec \\
  -B ${{worker_space_dir}}:/srv/ \\
  -B dask-credentials:/etc/dask-credentials \\
  {image_name} \\
  dask worker \\
    --name $2 \\
    --tls-ca-file /etc/dask-credentials/dask.crt \\
    --tls-cert   /etc/dask-credentials/dask.crt \\
    --tls-key    /etc/dask-credentials/dask.pem \\
    --worker-port 10000:10070 \\
    --no-nanny \\
    --local-directory /srv \\
    --scheduler-sni daskgateway-{cluster_name} \\
    --nthreads 1 \\
    tls://{self.scheduler_proxy_ip}:80
"""
        start_sh_path = f"{tmpdir}/start.sh"
        with open(start_sh_path, "w") as f:
            f.write(start_sh)
        os.chmod(start_sh_path, 0o755)

        # ------------------------------------------------------------------
        # Step 4 — submit via jobsub_submit
        # ------------------------------------------------------------------
        # jobsub_lite already sets transfer_output_files = .empty_file in its
        # template (simple.cmd line 26) so we do NOT include it in lines_flags.
        lines_flags = [
            "--lines",
            "should_transfer_files = yes",
            "--lines",
            (f"transfer_input_files = {credentials_dir},{worker_space_dir}"),
            "--lines",
            "when_to_transfer_output = ON_EXIT_OR_EVICT",
            "--lines",
            "+isDaskJob = True",
        ]
        if self.node_match_expr:
            lines_flags += ["--lines", self.node_match_expr]

        # file:// URI scheme is required by jobsub_lite's CheckExecutable.
        # tmpdir starts with '/', so f"file://{tmpdir}/start.sh" produces
        # the correct triple-slash form: file:///tmp/htcdask-.../start.sh
        cmd_parts = [
            "jobsub_submit",
            "-G",
            self.jobsub_group,
            "--memory",
            worker_mem,
            "--cpu",
            num_cores,
            "-N",
            str(n),
            *lines_flags,
            f"file://{tmpdir}/start.sh",
            # Positional args become $1 and $2 in start.sh.
            # $(Cluster) and $(Process) are HTCondor submit macros; they are
            # passed as literal strings here (no shell involved) and written
            # verbatim into the 'arguments' line of the .cmd file by
            # jobsub_lite, where HTCondor expands them at submit time.
            cluster_name,
            "htcdask-worker_$(Cluster)_$(Process)",
        ]

        logger.info("Sandbox: %s", tmpdir)
        logger.info("Using image: %s", image_name)
        logger.debug("Submitting %d jobsub worker(s)", n)

        output = subprocess.check_output(cmd_parts, cwd=tmpdir)

        # ------------------------------------------------------------------
        # Step 5 — parse job ID from jobsub_submit output
        # Expected line: "Use job id 12345678.0@schedd.example.com to retrieve output"
        # Regex from jobsub_lite/lib/jobsub_api.py
        # ------------------------------------------------------------------
        m = re.search(
            r"Use job id (\d+)\.\d+@(\S+) to retrieve",
            output.decode(),
        )
        if not m:
            raise RuntimeError(
                f"Could not parse job ID from jobsub_submit output:\n{output.decode()}"
            )
        clusterid = m.group(1)  # e.g. "12345678"
        scheddname = m.group(2)  # e.g. "schedd.example.com"

        logger.info(
            "Submitted %d HTCondor job(s) to %s with ClusterId %s",
            n,
            scheddname,
            clusterid,
        )

        return {
            "ClusterId": clusterid,
            "Iwd": tmpdir,
            "ScheddName": scheddname,
            "n_jobs": n,
        }

    # ------------------------------------------------------------------
    # Teardown helpers
    # ------------------------------------------------------------------

    def _destroy_batch_cluster(self, cluster):
        """Remove an entire HTCondor cluster via ``jobsub_rm``."""
        try:
            cmd = [
                "jobsub_rm",
                "-G",
                self.jobsub_group,
                "-J",
                f"{cluster['ClusterId']}@{cluster['ScheddName']}",
            ]
            result = subprocess.check_output(cmd)
            logger.info(
                "Removed cluster %s: %s",
                cluster["ClusterId"],
                result.decode().rstrip(),
            )
        except Exception as e:
            logger.error(
                "Failed to remove cluster %s: %s",
                cluster.get("ClusterId"),
                e,
            )

    def _remove_jobs_in_cluster(self, cluster, n):
        """Remove ``n`` individual jobs from a cluster via ``jobsub_rm``.

        Queries proc IDs from the schedd, then removes the first ``n`` via a
        single ``jobsub_rm -J id1,id2,...`` call (the ``-J`` flag accepts a
        comma-separated list and splits internally).
        """
        q_cmd = [
            "condor_q",
            cluster["ClusterId"],
            "-name",
            cluster["ScheddName"],
            "-af",
            "ProcId",
        ]
        result = subprocess.check_output(q_cmd)
        proc_ids = [int(x) for x in result.decode().split()]

        if n > len(proc_ids):
            logger.error(
                "Requested to remove %d jobs but only %d found; "
                "removing entire cluster",
                n,
                len(proc_ids),
            )
            self._destroy_batch_cluster(cluster)
            return

        job_ids = ",".join(
            f"{cluster['ClusterId']}.{pid}@{cluster['ScheddName']}"
            for pid in proc_ids[:n]
        )
        rm_cmd = ["jobsub_rm", "-G", self.jobsub_group, "-J", job_ids]
        subprocess.check_output(rm_cmd)
        logger.info(
            "Removed %d job(s) from cluster %s",
            n,
            cluster["ClusterId"],
        )
        cluster["n_jobs"] -= n

    def destroy_all_batch_clusters(self):
        """Remove all tracked HTCondor clusters via ``jobsub_rm``."""
        logger.info("Shutting down jobsub worker jobs (if any)")
        if not self.batchWorkerJobs:
            return
        for htc_cluster in self.batchWorkerJobs:
            try:
                cmd = [
                    "jobsub_rm",
                    "-G",
                    self.jobsub_group,
                    "-J",
                    f"{htc_cluster['ClusterId']}@{htc_cluster['ScheddName']}",
                ]
                result = subprocess.check_output(cmd)
                logger.info(" %s", result.decode().rstrip())
            except Exception as e:
                logger.error(
                    "Failed to remove jobsub cluster %s: %s",
                    htc_cluster.get("ClusterId"),
                    e,
                )
        self.batchWorkerJobs = []

    async def _stop_async(self):
        """Shut down the cluster and clean up local temp directories.

        Call chain:
            JobsubGatewayCluster._stop_async()
              └─ super()._stop_async()  →  HTCGatewayCluster._stop_async()
                   ├─ self.destroy_all_batch_clusters()   (dynamic dispatch → ours)
                   └─ await super()._stop_async()  →  GatewayCluster._stop_async()
              └─ shutil.rmtree for each tmpdir in self._tmpdirs
        """
        await super()._stop_async()
        for tmpdir in self._tmpdirs:
            if os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
        self._tmpdirs = []
