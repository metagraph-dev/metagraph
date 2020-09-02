import os
import uuid
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


with open(
    os.path.join(os.path.dirname(__file__), "./templates/page.html"), "r"
) as file_handle:
    PAGE_HTML = file_handle.read()

with open(
    os.path.join(os.path.dirname(__file__), "./templates/app.html"), "r"
) as file_handle:
    APP_HTML = file_handle.read()

with open(
    os.path.join(os.path.dirname(__file__), "./templates/shadow.css"), "r"
) as file_handle:
    SHADOW_CSS = file_handle.read()

with open(
    os.path.join(os.path.dirname(__file__), "./templates/shadow.html"), "r"
) as file_handle:
    SHADOW_HTML = file_handle.read()


def render_text(resolver, port, div=None):
    if div is None:
        unique_id = uuid.uuid4().int
        div = f"RandomDiv_{unique_id}"

    resolver_json = json.dumps(
        {
            "divId": div,
            "port": port,
            "shadowInnerHTML": f'<style type="text/css">{SHADOW_CSS}</style>{SHADOW_HTML}',
            # Eagerly store select API results
            "abstractTypeToConcreteTypes": api.list_types(resolver),
            "pluginData": api.get_plugins(resolver),
            "abstractTypes": api.get_abstract_types(resolver),
        }
    )

    return APP_HTML.replace("{DIV_ID}", div).replace("{RESOLVER_DATA}", resolver_json)


def write_tempfile(text):
    f = tempfile.NamedTemporaryFile(suffix=".html")
    page_text = PAGE_HTML.replace("{BODY}", text)
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
                message = json.dumps(
                    {"function": func, "result": result, "input_kwargs": kwargs}
                )
                await asyncio.wait(
                    [conn.send(message) for conn in self.active_connections]
                )
        finally:
            await self.unregister(websocket)


def main(resolver, embedded=True):
    if embedded and not has_nest_asyncio:
        print("nest_asyncio is required to use the explorer from within a notebook")
        embedded = False

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

        # import panel
        # panel.extension()

        text = render_text(resolver, port)
        return HTML(text)
        # return panel.pane.markup.HTML(text, style={'width': '100%'}, sizing_mode='stretch_both')
    else:
        text = render_text(resolver, port, "mgExplorer")
        f = write_tempfile(text)
        webbrowser.open(f"file://{f.name}")
        asyncio.get_event_loop().run_forever()
