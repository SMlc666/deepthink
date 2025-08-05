import asyncio
import json
import queue
import threading
from flask import Flask, render_template, request, Response, jsonify
from src.orchestrator import Orchestrator
from src import history_manager
from starlette.middleware.wsgi import WSGIMiddleware

# The Flask app instance is now named 'flask_app'
flask_app = Flask(__name__)

# 确保模板和静态文件在修改后能立即生效
flask_app.config['TEMPLATES_AUTO_RELOAD'] = True

@flask_app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@flask_app.route('/history', methods=['GET'])
def get_history():
    """获取历史记录"""
    history = history_manager.load_history()
    return jsonify(history)

@flask_app.route('/run', methods=['GET'])
def run_orchestrator():
    """
    接收前端请求，启动Orchestrator，并通过Server-Sent Events (SSE)流式传输结果。
    This view is synchronous, but it starts an async task in a background thread.
    """
    problem = request.args.get('problem')
    num_solutions = int(request.args.get('solutions', 3))
    mode = request.args.get('mode', 'batch')
    verbose = request.args.get('verbose', 'true').lower() == 'true'

    if not problem:
        return Response(json.dumps({'error': '问题不能为空'}), status=400, mimetype='application/json')

    # Create history entry
    history_entry = history_manager.add_history_entry(
        problem=problem,
        mode=mode,
        solutions=num_solutions,
        source='web'
    )
    history_id = history_entry['id']

    # A thread-safe queue to pass data from the async thread to the sync Flask thread
    q = queue.Queue()
    all_updates = []

    async def producer():
        """
        异步生产者，运行Orchestrator并将结果放入队列。
        """
        try:
            orchestrator = Orchestrator(verbose=verbose)
            async for update in orchestrator.run(
                problem=problem,
                num_solutions=num_solutions,
                mode=mode,
                history_id=history_id
            ):
                all_updates.append(update)
                sse_message = f"data: {json.dumps(update)}\n\n"
                q.put(sse_message)
                await asyncio.sleep(0.01)
        except Exception as e:
            import traceback
            print(f"Error during orchestration: {e}")
            traceback.print_exc()
            error_message = {"event": "error", "nodeId": "root", "content": f"发生错误: {str(e)}"}
            sse_message = f"data: {json.dumps(error_message)}\n\n"
            q.put(sse_message)
            history_manager.update_history_entry(history_id, {"status": "failed", "graph_data": all_updates})
        finally:
            # This block will run whether there was an error or not
            # Find the entry and update it with the final graph data
            history = history_manager.load_history()
            entry_found = False
            for entry in history:
                if entry.get("id") == history_id:
                    # We only update the graph_data here.
                    # The status/final_review is updated in the orchestrator.
                    entry["graph_data"] = all_updates
                    entry_found = True
                    break
            if entry_found:
                history_manager.save_history(history)

            # Signal the end of the stream
            q.put(None)

    def run_async_producer():
        """Helper function to run the async producer in a new event loop."""
        asyncio.run(producer())

    # Start the async producer in a background thread
    thread = threading.Thread(target=run_async_producer)
    thread.start()

    def generate_sync():
        """
        同步生成器，从队列中获取数据并yield给Flask。
        This runs in the Flask thread.
        """
        while True:
            item = q.get()  # This will block until an item is available
            if item is None:
                break
            yield item

    return Response(generate_sync(), mimetype='text/event-stream')


# Wrap the Flask app with the WSGIMiddleware
# Uvicorn will now correctly interact with this 'app' object
app = WSGIMiddleware(flask_app)

if __name__ == '__main__':
    # For local debugging, run the original Flask app
    # For production, it's recommended to use: uvicorn app:app --reload
    flask_app.run(debug=True, port=5000)
