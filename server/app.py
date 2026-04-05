"""OpenEnv entry: validator requires server/app.py with def main(...) and if __name__ + main()."""

import uvicorn


def main(host: str = "0.0.0.0", port: int = 7860):
    from app import app as fastapi_app

    uvicorn.run(fastapi_app, host=host, port=port)


if __name__ == "__main__":
    main()
