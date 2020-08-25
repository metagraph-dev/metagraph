import asyncio
import string
import random
import websockets
import webbrowser
import tempfile
import socket
import errno
import json
from . import api
from .. import config

try:
    import nest_asyncio

    has_nest_asyncio = True
    nest_asyncio.apply()
except ImportError:
    has_nest_asyncio = False


page_html = r"""
<!DOCTYPE html>
<html>
<head>
    <title>Metagraph Explorer</title>
</head>
<body>
{BODY}
</body>
</html>
"""

app_html = r"""
    <div id="{DIV_ID}" />
    <script>
        var abstractTypesFor{DIV_ID} = {ABSTRACT_TYPES};
        var pluginsFor{DIV_ID} = {PLUGINS};
        
        var div = document.querySelector('#{DIV_ID}');
        var shadow = div.attachShadow({mode: 'open'});
        shadow.innerHTML = `
            <style type="text/css">
                body {
                    font-family: "Courier New", sans-serif;
                    text-align: center;
                }
                
                /* Remove default bullets */
                .treewidget ul {
                  list-style-type: none;
                }
                
                /* Style the caret */
                .treewidget .caret {
                  cursor: pointer;
                  user-select: none; /* Prevent text selection */
                }
                
                /* Create the caret */
                .treewidget .caret::before {
                  content: "\u25B6";
                  color: black;
                  display: inline-block;
                  margin-right: 6px;
                }
                
                /* Rotate the caret when clicked */
                .treewidget .caret-down::before {
                  transform: rotate(90deg);
                }
                
                /* Hide the nested list */
                .treewidget .nested {
                  display: none;
                }
                
                /* Show the nested list when the user clicks on the caret */
                .treewidget .active {
                  display: block;
                }
            </style>
            
            <div>
              <button class="close button">Close</button>
              <div>
                <input type="radio" id="viewTypeTypes1"
                 name="viewType" value="types">
                <label for="viewTypeTypes1">Types</label>
            
                <input type="radio" id="viewTypeTranslators2"
                 name="viewType" value="translators">
                <label for="viewTypeTranslators2">Translators</label>
            
                <input type="radio" id="viewTypeAlgorithms3"
                 name="viewType" value="algorithms">
                <label for="viewTypeAlgorithms3">Algorithms</label>
              </div>
              <div>
                <label for="pluginsDropDown">Filter by Plugin</label>
                <select name="plugins" id="pluginsDropDown">
                  <option name="">---</option>
                </select>
              </div>
              <div style="visibility: hidden">
                <label for="abstractDropDown">Source Type</label>
                <select name="abstractTypes" id="abstractDropDown"></select>
              </div>
              <div id="display"></div>
            </div>
        `;

        var buildTree = function(root, data, topLevel) {
            if (topLevel) {
                root.innerHTML = "";
                var div = document.createElement('div');
                div.className = 'treewidget';
                root.appendChild(div);
                var ul = document.createElement('ul');
                div.appendChild(ul);
                buildTree(ul, data, false);
                
                var toggler = root.getElementsByClassName("caret");
                for (var i = 0; i < toggler.length; i++) {
                  toggler[i].addEventListener("click", function() {
                    this.parentElement.querySelector(".nested").classList.toggle("active");
                    this.classList.toggle("caret-down");
                  });
                }
            } else {
                for (var name in data) {
                    if (data.hasOwnProperty(name)) {
                        var li = document.createElement('li');
                        var props = data[name];
                        if ('children' in props) {
                            var caret = document.createElement('span');
                            caret.className = 'caret';
                            caret.innerHTML = name;
                            li.appendChild(caret);
                            var ul = document.createElement('ul');
                            ul.className = 'nested';
                            li.appendChild(ul);
                            root.appendChild(li);
                            buildTree(ul, props['children'], false);
                        } else {
                            li.innerHTML = name;
                            root.appendChild(li);
                        }
                    }
                }
            }
        }

        shadow.close = shadow.querySelector(".close");
        shadow.viewTypes = shadow.querySelector("#viewTypeTypes1");
        shadow.viewTranslators = shadow.querySelector("#viewTypeTranslators2");
        shadow.viewAlgorithms = shadow.querySelector("#viewTypeAlgorithms3");
        shadow.pluginsDropDown = shadow.querySelector('#pluginsDropDown');
        shadow.abstractDropDown = shadow.querySelector('#abstractDropDown');
        shadow.display = shadow.querySelector("#display");
        shadow.websocket = new WebSocket("ws://127.0.0.1:{PORT}/");
        // Add back-reference on websocket
        shadow.websocket.owner = shadow;

        pluginsFor{DIV_ID}.forEach(function(plug) {
            var op = document.createElement('option');
            op.value = plug;
            op.innerHTML = plug;
            shadow.pluginsDropDown.appendChild(op);
        });
        shadow.pluginsDropDown.onchange = function(event) {
            allHandler(this.getRootNode());
        }
        
        abstractTypesFor{DIV_ID}.forEach(function(at) {
            var op = document.createElement('option');
            op.value = at;
            op.innerHTML = at;
            shadow.abstractDropDown.appendChild(op);
        });
        shadow.abstractDropDown.onchange = function(event) {
            allHandler(this.getRootNode());
        }

        var radioHandler = function (event) {
            var shadow = this.getRootNode();
            if (shadow.viewTranslators.checked) {
                shadow.abstractDropDown.parentElement.style.visibility = null;
            } else {
                shadow.abstractDropDown.parentElement.style.visibility = "hidden";
            }
            allHandler(shadow);
        }
        shadow.viewTypes.onchange = radioHandler;
        shadow.viewTranslators.onchange = radioHandler;
        shadow.viewAlgorithms.onchange = radioHandler;
        shadow.close.onclick = function (event) {
            root = this.getRootNode();
            root.websocket.send(JSON.stringify({function: "close"}));
            root.websocket.close();
            // Remove everything as part of cleanup
            root.innerHTML = "";
        }
        
        var allHandler = function(shadow) {
            var kwargs = {filters: {}};
            var pluginFilter = shadow.pluginsDropDown.value;
            if (pluginFilter != "---") {
                kwargs['filters']['plugin'] = pluginFilter;
            }
        
            if (shadow.viewTypes.checked) {
                shadow.display.innerHTML = '';
                shadow.websocket.send(JSON.stringify({function: "list_types", kwargs: kwargs}));
            } else if (shadow.viewTranslators.checked) {
                shadow.display.innerHTML = '';
                kwargs['source_type'] = shadow.abstractDropDown.value;
                shadow.websocket.send(JSON.stringify({function: "list_translators", kwargs: kwargs}));
            } else if (shadow.viewAlgorithms.checked) {
                shadow.display.innerHTML = '';
                shadow.websocket.send(JSON.stringify({function: "list_algorithms", kwargs: kwargs}));
            }
        }
        
        shadow.websocket.onmessage = function (event) {
            shadow = this.owner;
            data = JSON.parse(event.data);
            switch (data.function) {
                case "list_types":
                    buildTree(shadow.display, data.result, true);
                    break;
                case "list_translators":
                    shadow.display.innerHTML = "";
                    var div = document.createElement('div');
                    div.className = "treewidget";
                    shadow.display.appendChild(div);
                    var ul = document.createElement('ul');
                    div.appendChild(ul);
                    var primary = document.createElement('li');
                    primary.innerHTML = "Primary: " + data.result['primary_types'];
                    ul.appendChild(primary);
                    var secondary = document.createElement('li');
                    secondary.innerHTML = "Secondary: " + data.result['secondary_types'];
                    ul.appendChild(secondary);
                    var trans = document.createElement('li');
                    ul.appendChild(trans);
                    var caret = document.createElement('span');
                    caret.className = 'caret caret-down';
                    caret.innerHTML = 'Translators'
                    trans.appendChild(caret);
                    var transUL = document.createElement('ul')
                    transUL.className = 'nested active';
                    trans.appendChild(transUL);
                    for (var i = 0; i < data.result['translators'].length; i++) {
                        var t = data.result['translators'][i];
                        var t2 = document.createElement('li');
                        t2.innerHTML = t;
                        transUL.appendChild(t2);
                    }
                    caret.addEventListener("click", function() {
                        this.parentElement.querySelector(".nested").classList.toggle("active");
                        this.classList.toggle("caret-down");
                    });
                    break;
                case "list_algorithms":
                    buildTree(shadow.display, data.result, true);
                    break;
                default:
                    console.error(
                        "unsupported event", data);
            }
        };
    </script>
"""


def render_text(resolver, port, div=None):
    if div is None:
        div = "RandomDiv_" + "".join(random.sample(string.ascii_letters, 16))

    abstract_types = json.dumps(api.get_abstract_types(resolver))
    plugins = json.dumps(api.list_plugins(resolver))

    return (
        app_html.replace("{PORT}", str(port))
        .replace("{DIV_ID}", div)
        .replace("{ABSTRACT_TYPES}", abstract_types)
        .replace("{PLUGINS}", plugins)
    )


def write_tempfile(text):
    f = tempfile.NamedTemporaryFile(suffix=".html")
    page_text = page_html.replace("{BODY}", text)
    f.write(page_text.encode("ascii"))
    f.flush()
    return f


def find_open_port(initial_port=5678):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = initial_port

    while True:
        try:
            s.bind(("127.0.0.1", port))
            break
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                if config.get("explorer.verbose", False):
                    print(f"Port {port} is already in use")
                port += 1
            else:
                # something else raised the socket.error exception
                if config.get("explorer.verbose", False):
                    print(e)

    s.close()
    return port


class Service:
    def __init__(self, resolver, port, embedded=True):
        self.resolver = resolver
        self.port = port
        self._embedded = embedded
        self.active_connections = set()
        self._is_running = True
        self.server = None  # This will be monkey-patched later

    async def register(self, websocket):
        self.active_connections.add(websocket)

    async def unregister(self, websocket):
        self.active_connections.remove(websocket)
        if not self.active_connections:
            self.server.close()
            if not self._embedded:
                asyncio.get_event_loop().stop()

    async def handler(self, websocket, path):
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                func = data["function"]
                if func == "close":
                    break
                kwargs = data.get("kwargs", {})
                result = getattr(api, func)(self.resolver, **kwargs)
                message = json.dumps({"function": func, "result": result,})
                await asyncio.wait(
                    [conn.send(message) for conn in self.active_connections]
                )
        finally:
            await self.unregister(websocket)


def main(resolver, embedded=True):
    if embedded and not has_nest_asyncio:
        raise ImportError(
            "nest_asyncio is required to use the explorer from within a notebook"
        )

    port = find_open_port()
    try:

        async def start_service():
            service = Service(resolver, port, embedded=embedded)
            server = await websockets.serve(service.handler, "127.0.0.1", port)
            service.server = server

        asyncio.get_event_loop().run_until_complete(start_service())
        if config.get("explorer.verbose", False):
            print(f"serving explorer on port {port}")
    except RuntimeError:
        import traceback

        traceback.print_exc()
        return

    if embedded:
        from IPython.core.display import HTML

        text = render_text(resolver, port)
        return HTML(text)
    else:
        text = render_text(resolver, port, "mgExplorer")
        f = write_tempfile(text)
        webbrowser.open(f"file://{f.name}")
        asyncio.get_event_loop().run_forever()
