from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uuid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--model', type=str, default="/home/gomall/models/DeepSeek-R1-Distill-Qwen-7B/")
parser.add_argument('--port', type=str, default="8003")
args = parser.parse_args()

app = FastAPI()

# Optimized engine parameters
engine_args = AsyncEngineArgs(
    model=args.model,
    tensor_parallel_size=1,
    max_model_len=2048,  # Limit the context length
    gpu_memory_utilization=0.9,  # Reserve 10% GPU memory
    dtype="float16",  # FP16 inference
    max_num_seqs=32,  # Maximum number of concurrent requests
    max_num_batched_tokens=2048,  # Maximum number of tokens per batch
    trust_remote_code=True,
    device=f"cuda:{args.device}"
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    max_tokens = data.get("max_tokens", 512)  # Default 512

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens
    )

    request_id = str(uuid.uuid4())
    results_generator = engine.generate(prompt, sampling_params, request_id)

    async for request_output in results_generator:
        final_output = request_output

    return JSONResponse({"response": final_output.outputs[0].text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
