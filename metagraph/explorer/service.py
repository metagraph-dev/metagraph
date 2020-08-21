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
        var div = document.querySelector('#{DIV_ID}');
        var shadow = div.attachShadow({mode: 'open'});
        shadow.innerHTML = `
            <style type="text/css">
                body {
                    font-family: "Courier New", sans-serif;
                    text-align: center;
                }
                .buttons {
                    font-size: 4em;
                    display: flex;
                    justify-content: center;
                }
                .button, .value {
                    line-height: 1;
                    padding: 2rem;
                    margin: 2rem;
                    border: medium solid;
                    min-height: 1em;
                    min-width: 1em;
                }
                .button {
                    cursor: pointer;
                    user-select: none;
                }
                .minus {
                    color: red;
                }
                .plus {
                    color: green;
                }
                .value {
                    min-width: 2em;
                }
                .state {
                    font-size: 2em;
                }
            </style>
            <div class="buttons">
                <div class="minus button">-</div>
                <div class="value">?</div>
                <div class="plus button">+</div>
            </div>
            <div class="close button">Close</div>
        `;

        shadow.minus = shadow.querySelector(".minus");
        shadow.plus = shadow.querySelector(".plus");
        shadow.value = shadow.querySelector(".value");
        shadow.close = shadow.querySelector(".close");
        shadow.websocket = new WebSocket("ws://127.0.0.1:{PORT}/");
        // Add back-reference on websocket
        shadow.websocket.owner = shadow;

        shadow.minus.onclick = function (event) {
            this.getRootNode().websocket.send(JSON.stringify({function: "minus"}));
        }
        shadow.plus.onclick = function (event) {
            this.getRootNode().websocket.send(JSON.stringify({function: "plus"}));
        }
        shadow.close.onclick = function (event) {
            root = this.getRootNode();
            root.websocket.send(JSON.stringify({function: "close"}));
            root.websocket.close();
            // Remove everything as part of cleanup
            root.innerHTML = "";
        }
        shadow.websocket.onmessage = function (event) {
            root = this.owner;
            data = JSON.parse(event.data);
            switch (data.type) {
                case "state":
                    root.value.textContent = data.value;
                    break;
                default:
                    console.error(
                        "unsupported event", data);
            }
        };
    </script>
"""


def write_tempfile(port):
    f = tempfile.NamedTemporaryFile(suffix=".html")
    app_text = app_html.replace("{PORT}", str(port)).replace("{DIV_ID}", "mgExplorer")
    page_text = page_html.replace("{BODY}", app_text)
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
    def __init__(self, resolver, port, ipython=False):
        self.resolver = resolver
        self.port = port
        self._ipython = ipython
        self.active_connections = set()
        self.valuable = api.Valuable()
        self._is_running = True
        self.server = None  # This will be monkey-patched later

    async def register(self, websocket):
        self.active_connections.add(websocket)

    async def unregister(self, websocket):
        self.active_connections.remove(websocket)
        if not self.active_connections:
            self.server.close()
            if not self._ipython:
                asyncio.get_event_loop().stop()

    async def handler(self, websocket, path):
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                func = data["function"]
                if func == "close":
                    break
                # ------------------
                if func == "minus":
                    self.valuable.adjust_value(-1)
                    await self.valuable.notify_state(self.active_connections)
                    continue
                if func == "plus":
                    self.valuable.adjust_value(+1)
                    await self.valuable.notify_state(self.active_connections)
                    continue
                # ------------------
                kwargs = data.get("kwargs", {})
                result = getattr(api, func)()
        finally:
            await self.unregister(websocket)

    async def counter(self, websocket, path):
        # register(websocket) sends user_event() to websocket
        await self.register(websocket)
        try:
            await websocket.send(self.valuable.state_event())
            async for message in websocket:
                data = json.loads(message)
                if data["action"] == "minus":
                    self.valuable.adjust_value(-1)
                    await self.valuable.notify_state(self.active_connections)
                elif data["action"] == "plus":
                    self.valuable.adjust_value(+1)
                    await self.valuable.notify_state(self.active_connections)
                elif data["action"] == "close":
                    break
        finally:
            await self.unregister(websocket)


def main(resolver, ipython=False):
    if ipython and not has_nest_asyncio:
        raise ImportError(
            "nest_asyncio is required to use the explorer from within a notebook"
        )

    port = find_open_port()
    try:

        async def start_service():
            service = Service(resolver, port, ipython=ipython)
            server = await websockets.serve(service.handler, "127.0.0.1", port)
            service.server = server

        asyncio.get_event_loop().run_until_complete(start_service())
        if config.get("explorer.verbose", False):
            print(f"serving explorer on port {port}")
    except RuntimeError:
        import traceback

        traceback.print_exc()
        return

    if ipython:
        from IPython.core.display import HTML

        rand_divname = "RandomDiv_" + "".join(random.sample(string.ascii_letters, 16))
        return HTML(
            app_html.replace("{PORT}", str(port)).replace("{DIV_ID}", rand_divname)
        )
    else:
        f = write_tempfile(port)
        webbrowser.open(f"file://{f.name}")
        asyncio.get_event_loop().run_forever()
