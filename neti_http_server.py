#import ConfigParser
import configparser as ConfigParser
import time
import cgi
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
#from src.neti_neti_trainer import NetiNetiTrainer
from src.neti_neti_trainer import NetiNetiTrainer
from src.neti_neti import NetiNeti
import json
config = ConfigParser.ConfigParser()
config.read('config/neti_http_config.cfg')
HOST = config.get('http_settings', 'host')
PORT = int(config.get('http_settings', 'port'))

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query_components = parse_qs(urlparse(self.path).query)
        input = query_components["input"]
        print('Input :',input)
        output = nn.find_names(str(input))
        print('Output :',output)
        #added by ASD
        if output:
            self.send_response(200, 'OK')
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            #self.wfile.write('hello world')
            self.wfile.write(bytes(json.dumps(output, ensure_ascii=False), 'utf-8'))
        else:
            self.send_response(400)
            self.end_headers()

    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     })
        self.send_response(200)
        self.end_headers()
        self.wfile.write(nn.find_names(form['data'].value))
        return

def run(server_class=HTTPServer,
        handler_class=MyHandler):
    server_address = (HOST, PORT)
    print('Starting server, use <Ctrl-C> to stop')
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

if __name__ == '__main__':
    print( "Running NetiNeti Training, it might take a while...")
    nnt = NetiNetiTrainer()
    nn = NetiNeti(nnt)
    run()
