import sys
import os

_PROJECT_PATH="/".join( os.path.abspath(__file__).split("/")[0:-2] )

if __name__ == "__main__":
    from streamlit import cli as stcli
    print("#####################")
    print("# Start Streamlit ! #")
    print("#####################")
    port=int(sys.argv[1])
    sys.argv= ["streamlit", "run", os.path.join(_PROJECT_PATH,"front/main.py"), f"--server.port={port}"]
    sys.exit(stcli.main())
    print("app: server stop...")