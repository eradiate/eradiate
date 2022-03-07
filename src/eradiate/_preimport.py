"""
Pre-import problematic modules.
"""

import os
import sys
import threading
import time


# See
# https://stackoverflow.com/questions/24277488/in-python-how-to-capture-the-stdout-from-a-c-shared-library-to-a-variable
class _OutputGrabber(object):
    """
    Class used to grab standard output or another stream.
    """

    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.threaded = threaded
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)

    def stop(self):
        """
        Stop capturing the stream data and save the text in `capturedtext`.
        """
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char


# Check if Dr.Jit and Mitsuba can be imported successfully
with _OutputGrabber(sys.stderr) as out:
    try:
        __import__("drjit")
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "Could not import module 'drjit'; did you build the kernel and add "
            "it to your $PYTHONPATH?"
        ) from e

    try:
        __import__("mitsuba")
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "Could not import module 'mitsuba'; did you build the kernel and add "
            "it to your $PYTHONPATH?"
        ) from e


# Silence a no-CUDA warning message (if any)
_warn_str = """
jit_cuda_init(): cuInit failed, disabling CUDA backend.
There are two common explanations for this type of failure:

 1. your computer simply does not contain a graphics card that supports CUDA.

 2. your CUDA kernel module and CUDA library are out of sync. Try to see if you
    can run a utility like 'nvida-smi'. If not, a reboot will likely fix this
    issue. Otherwise reinstall your graphics driver.

 The specific error message produced by cuInit was
   "no CUDA-capable device is detected"
"""
if _warn_str.strip() != out.capturedtext.strip():
    print(out.capturedtext)
del out
