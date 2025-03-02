import subprocess
import re
import time

class LocalTunnelService:
    def __init__(self, port=8080, subdomain="elderlycare"):
        self.port = port
        self.subdomain = subdomain
        self.process = None
        self.tunnel_url = None

    def start_local_tunnel(self):
        """
        Start localtunnel using the given port and subdomain.
        Returns the public URL once it is detected.
        """
        command = f"lt --port {self.port} --subdomain {self.subdomain}"
        # Start the tunnel as a subprocess.
        self.process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Try to capture the public URL from stdout.
        start_time = time.time()
        while True:
            # Timeout after 30 seconds if no URL is found.
            if time.time() - start_time > 30:
                stderr_output = self.process.stderr.read()
                raise Exception(f"Localtunnel did not start within 30 seconds.\nStderr: {stderr_output}")

            # Read one line from stdout.
            line = self.process.stdout.readline()
            if line:
                print("LocalTunnel output:", line.strip())
                # Use a regex to find a URL starting with "https://"
                match = re.search(r"https:\/\/[^\s]+", line)
                if match:
                    self.tunnel_url = match.group(0)
                    break
            else:
                time.sleep(0.1)
        return self.tunnel_url

    def get_tunnel_url(self):
        """
        Returns the public tunnel URL. If the tunnel is not started yet, start it.
        """
        if self.tunnel_url is None:
            return self.start_local_tunnel()
        return self.tunnel_url

    def stop_tunnel(self):
        """
        Terminate the localtunnel process.
        """
        if self.process:
            self.process.terminate()
            self.process = None
            self.tunnel_url = None

# Example usage:
if __name__ == "__main__":
    tunnel_service = LocalTunnelService(port=8080, subdomain="elderlycare")
    try:
        public_url = tunnel_service.get_tunnel_url()
        print(f"LocalTunnel URL: {public_url}")
        # Keep the script running (e.g., for testing) until interrupted.
        while True:
            time.sleep(1)
    except Exception as e:
        print("Error:", e)
    finally:
        tunnel_service.stop_tunnel()
