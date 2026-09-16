"""Manage an isolated native Isaac process; no Isaac packages enter MuJoCo Python."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import fcntl
from backends.config import isaac_installation


class IsaacWorker:
    def __init__(self, output: Path, gpu: int = 6):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        lock_dir=Path(__file__).resolve().parents[2]/'temp/backend_parity'
        lock_dir.mkdir(parents=True,exist_ok=True)
        self.gpu_lock=(lock_dir/f'gpu-{gpu}.lock').open('a')
        try:fcntl.flock(self.gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.gpu_lock.close()
            raise RuntimeError(f'GPU {gpu} already has a Hooke Isaac job running')
        self.directory=tempfile.TemporaryDirectory(prefix='hooke-isaac-')
        self.socket_path=Path(self.directory.name)/'worker.sock'
        self.stream=None;self.connection=None;self.process=None
        memory=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if memory >= 2048:
            self.directory.cleanup();self.gpu_lock.close()
            raise RuntimeError(f'GPU {gpu} is busy ({memory} MiB); wait for the running job')
        uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        env=os.environ.copy()
        for key in ('CONDA_PREFIX','PYTHONPATH','PYTHONHOME','PYTHONEXE'):env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES=uuid,HOOKE_ISAAC_RENDER_GPU=str(gpu),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        install=isaac_installation()
        self.log=(self.output/'isaac-worker.log').open('wb')
        self.process=subprocess.Popen([str(install/'python.sh'),str(Path(__file__).with_name('isaac_worker.py')),'--socket',str(self.socket_path)],env=env,stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while not self.socket_path.exists():
                if self.process.poll() is not None:raise RuntimeError(f'Isaac worker exited; inspect {self.output / "isaac-worker.log"}')
                if time.monotonic()>deadline:raise TimeoutError('Isaac startup exceeded 180 seconds')
                time.sleep(.1)
            self.connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            self.connection.settimeout(180)
            self.connection.connect(str(self.socket_path))
            self.stream=self.connection.makefile('rwb')
        except BaseException:
            self.close();raise

    def call(self, op: str, **kwargs):
        self.stream.write((json.dumps({'op':op,**kwargs},allow_nan=False)+'\n').encode());self.stream.flush()
        line=self.stream.readline(16*1024*1024)
        if not line:raise RuntimeError('Isaac worker disconnected; inspect its log')
        response=json.loads(line)
        if not response['ok']:raise RuntimeError(response['error']+'\n'+response.get('traceback',''))
        return response['result']

    def close(self):
        if self.stream is not None:
            try:self.call('shutdown')
            except (OSError,ValueError,RuntimeError):pass
            self.stream.close();self.stream=None
        if self.connection is not None:self.connection.close();self.connection=None
        if self.process is not None:
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=10)
        if hasattr(self,'log'):self.log.close()
        self.directory.cleanup()
        if hasattr(self,'gpu_lock'):self.gpu_lock.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
