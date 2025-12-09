from dotenv import load_dotenv
load_dotenv()
from app import create_app
import os

app = create_app()   # <-- tạo app từ factory

def main() -> None:
    app = create_app()
    host = os.getenv('MICRO_HOST', '0.0.0.0')
    port = int(os.getenv('MICRO_PORT', 5000))
    debug = os.getenv('MICRO_DEBUG', '0') in ('1', 'true', 'yes')

    print(f"Starting Flask microservice on {host}:{port}")
    print(f"Starting Flask service on {host}:{port}")
    app.run(host=host, port=port, debug=debug)



if __name__ == '__main__':
    main()