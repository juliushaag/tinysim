from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


WEB_PATH = Path(__file__).parent.resolve()

WEB_FOLDER = (Path(__file__).parent).resolve() / "res"
WEB_FILES = set(str(file.relative_to(WEB_FOLDER)) for file in WEB_FOLDER.glob("**/*.*"))


FILE_TYPES = {
  ".html" : 'text/html',
  ".css" : "style/css",
  ".js" : "text/javascript"
}

class WebRequestHandler(BaseHTTPRequestHandler):

  protocol_version = 'HTTP/1.1'
  on_data = lambda x: None

  def do_GET(self):
    # Handle GET requests
    parsed_path = urlparse(self.path)
    url_path = parsed_path.path
    
    url_path = url_path.strip()
    
    res = self._on_path_request(url_path)
    
    if res:
      resp, resp_type = res
      return self.respond(200, resp, resp_type)
    else:
      return self.respond(404, "")

  def do_POST(self):
    assert False


  def respond(self, status_code = 200, response = "", type = 'text/html'):
     
    self.send_response(status_code)
    self.send_header('Content-type', type)
    self.send_header('Content-Length', str(len(response)))
    self.end_headers()

    self.wfile.write(response.encode() if isinstance(response, str) else response)

  def log_message(self, format, *args):
    ... 

  def _on_path_request(self, whole_path : str):
    whole_path = whole_path[1:]
    path, content = whole_path.split("/", 1) if "/" in whole_path else ("", None)

    if path == "res" or path == "":
      content = content or "index.html"
      if content not in WEB_FILES: return None
      
      url_path = WEB_FOLDER / content
      html_content = url_path.read_text()
      type = "text/html"
      if url_path.suffix in FILE_TYPES: type = FILE_TYPES[url_path.suffix]
      return html_content, type 
    if path == "data":
      data = self.on_data(content)
      if data is None: return None
      return data, 'blob/bin'


WEB_SERVER = None
WEB_SERVER_THREAD = None
def start_web_server(host, port): 
  # mostly implemented for usage in jupyter notebooks
  if WEB_SERVER is not None: return
  global WEB_SERVER
  WEB_SERVER = ThreadingHTTPServer((host, port), WebRequestHandler)
  WEB_SERVER_THREAD = Thread(target=WEB_SERVER.serve_forever)
  WEB_SERVER_THREAD.start()
  