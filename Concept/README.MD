# Conceptualization Framework

This project is designed to generate concepts for the outputs of Named Entity Recognition (NER) and Event Detection (ED) tasks. It processes identified entities, creates prompts for a Large Language Model (LLM), and invokes the model to obtain conceptualization results.

## Quick Start

### 1. Install Dependencies

First, ensure you have installed all the necessary Python dependencies. Run the following command in the project's root directory:

```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing

This step converts the results from upstream tasks (like NER or ED) into an input format for the conceptualization task. The script generates a structured prompt query for each identified entity.

-   **Process NER Data:**
    ```bash
    python process_ner.py
    ```

-   **Process ED Data:**
    ```bash
    python process_ed.py
    ```

After running, these scripts will produce output files containing prompt queries, which will be used in the next step.

### 3. Run Conceptualization Inference

You can perform conceptualization inference in two ways: using a locally deployed large model service or by calling an external API directly.

#### Option A: Start a Local Model Service (Optional)

If you plan to use a local Large Language Model like Qwen for inference, you need to start an inference server first.

**Note:** This step requires you to have the model files and corresponding hardware environment ready.

```bash
python server.py
```

This script will start an HTTP service that waits for inference requests from the client.

#### Option B: Call an API (Skip Step 3)

If your `client.py` is configured to directly call a commercial or private LLM API (e.g., OpenAI, Anthropic), you can skip the step of starting the local `server.py`.

### 4. Execute Client Inference

Finally, run the client script to send requests and get the conceptualization results. The client reads the prompt query files generated in step 2 and sends them to the model for inference.

```bash
python client.py
```

After the inference is complete, the conceptualized results will be saved to the specified output files.

---
