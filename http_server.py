#!/usr/bin/env python
# -*- coding: utf-8 -*-


from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
from base64 import decodestring
import threading
import cgi
import cStringIO
import facedb
import utils
import urlparse
import json
import sys

fdb = None
fp = None

methods = set(['predict', 'list', 'delete', 'add'])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # hér er hægt að hafa fleiri aðgerðir, t.d. fá lista yfir id í grunni, lista yfir myndir per id og 
        # birta myndirnar sem eru til fyrir id (senda eina mynd sem inline base64 encode src í mg)
        parsed_path = urlparse.urlparse(self.path)
        print parsed_path
        self.send_response(200)
        self.end_headers()
        message = threading.currentThread().getName()
        self.wfile.write(message)
        self.wfile.write('\n')
        return

    def do_POST(self):
        # get the image from a parameter, send image to face processing, return answer
        parsed_path = urlparse.urlparse(self.path)
        print parsed_path
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     })
        # Begin the response
        self.send_response(200)
        self.end_headers()
        # self.wfile.write('Client: %s\n' % str(self.client_address))
        # self.wfile.write('User-agent: %s\n' % str(self.headers['user-agent']))
        # self.wfile.write('Path: %s\n' % self.path)
        # self.wfile.write('Form data:\n')


        # todo: taka við skipunum, sem eru þá annað hvort í slóðinni eða hluti af gögnunum
        # default er að fletta upp andliti, en það getur líka verið málið að bæta andliti í safnið
        if "face" in form:
            facedata = form["face"].value
            # print facedata[:100]
            # convert facedata to an image... and do something with the image. First we can just save it
            data = decodestring(facedata)
            face_image = utils.base64png_to_image(data)
            if "process" in parsed_path.path:
                print "process face"
                face_image = fp.process(face_image)
                if face_image is None:
                    print "Error processing face, found nothing!"
                    self.wfile.write(json.dumps({"result": -1, "error": "No face found in image"}))
                    return

            res = fdb.find_face(face_image)
            print "returning: ", res
            # self.wfile.write('{{"result": {} }}'.format(res))
            self.wfile.write(json.dumps({"result": res}))
            # now read the image from image_output into opencv
            return
        else:
            # self.wfile.write('{"result": -1,\n')
            # self.wfile.write('"Error": "No image received" }')
            self.wfile.write(json.dumps({"result": -1, "error": "No image received"}))
            return

        '''
        # Echo back information about what was posted in the form
        for field in form.keys():
            field_item = form[field]
            if field_item.filename:
                # The field contains an uploaded file
                file_data = field_item.file.read()
                file_len = len(file_data)
                del file_data
                self.wfile.write('\tUploaded %s as "%s" (%d bytes)\n' %
                                 (field, field_item.filename, file_len))
            else:
                # Regular form value
                self.wfile.write('\t%s=%s\n' % (field, form[field].value))
        return
        '''


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


if __name__ == '__main__':

    host = '0.0.0.0'
    path = "/Users/matti/Dropbox/Skjöl/Meistaraverkefni/server/tmp5"
    port = 8080
    feature_method = "LPQ"
    metric = "chi"
    # þetta eru eiginlega nógu margir parametrar til að setja almennilega upp, en látum þetta duga
    if len(sys.argv) > 1: 
        path = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    if len(sys.argv) > 3:
        feature_method = sys.argv[3]
    if len(sys.argv) > 4:
        metric = sys.argv[4]

    print("Starting server on port {}, using db at [{}]".format(port, path))

    fdb = facedb.FaceDatabase(path, feature_method, metric)
    fp = utils.FaceProcessor()

    server = ThreadedHTTPServer((host, port), Handler)
    print 'Starting server, use <Ctrl-C> to stop'
    server.serve_forever()
