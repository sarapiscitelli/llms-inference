make build

#tail -f /dev/null
uvicorn --factory llama_cpp.server.app:create_app --host $HOST --port $PORT