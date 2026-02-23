"""Launch the visual workflow editor.

Usage:
    python -m fsm_agent_flow.editor [--port PORT] [--no-open] [--file PATH]
"""

import argparse
import sys
import webbrowser

from .server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="fsm-agent-flow visual editor")
    parser.add_argument("--port", type=int, default=8742, help="Server port (default: 8742)")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--file", type=str, default=None, help="Workflow JSON file to open")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    if args.file:
        url += f"?file={args.file}"

    print(f"Starting fsm-agent-flow editor on {url}")

    if not args.no_open:
        webbrowser.open(url)

    try:
        run_server(port=args.port, initial_file=args.file)
    except KeyboardInterrupt:
        print("\nEditor stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
