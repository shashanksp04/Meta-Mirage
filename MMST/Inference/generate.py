import sys
sys.path.append('../')
from chat_models.OpenAI_Chat import OpenAI_Chat
from chat_models.Client import Client
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse
import time
from rag_agent.main import MainAgent

# Global RAG worker function for multiprocessing
def rag_worker_process(rag_queue, result_dict, test_model, embed_model_name, device, api_base):
    """Separate process that handles all RAG requests to avoid multiple model loads"""
    import asyncio
    try:
        # Use default port 11434 if api_base is empty
        if not api_base or api_base == "":
            api_base = "http://127.0.0.1:11434/v1"
        rag_agent = MainAgent(test_model=test_model, embed_model_name=embed_model_name, device=device, api_base=api_base)
        rag_runner = rag_agent.main()
        
        # Create event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while True:
            request = rag_queue.get()
            if request is None:  # Poison pill to stop worker
                break
            
            item_id, query = request
            try:
                # Clear previous web search calls for this item
                rag_agent.web_search_calls = []
                # run_debug is async, so we need to await it
                rag_response = loop.run_until_complete(rag_runner.run_debug(query))
                # Extract agent's answer (everything after "Rag_Agent > ")
                # The response may contain the full conversation, so we need to extract the last agent response
                rag_answer = None
                if isinstance(rag_response, str):
                    # Find the last occurrence of "Rag_Agent > " or "Rag_Agent>"
                    last_agent_pos = -1
                    marker = None
                    
                    # Try to find the last "Rag_Agent > " or "Rag_Agent>"
                    if "Rag_Agent > " in rag_response:
                        last_agent_pos = rag_response.rfind("Rag_Agent > ")
                        marker = "Rag_Agent > "
                    elif "Rag_Agent>" in rag_response:
                        last_agent_pos = rag_response.rfind("Rag_Agent>")
                        marker = "Rag_Agent>"
                    
                    if last_agent_pos >= 0 and marker:
                        # Extract everything after the marker
                        start_pos = last_agent_pos + len(marker)
                        remaining_text = rag_response[start_pos:].strip()
                        
                        # Find where the next section starts (### or User >)
                        # Try multiple delimiter patterns
                        next_section_pos = len(remaining_text)
                        delimiters = [
                            "\n ### Continue session",
                            "\n### Continue session", 
                            "\n ###",
                            "\n###",
                            "\nUser >",
                            "\nUser>",
                            "\n Continue session",
                            "\nContinue session"
                        ]
                        for delimiter in delimiters:
                            pos = remaining_text.find(delimiter)
                            if pos >= 0 and pos < next_section_pos:
                                next_section_pos = pos
                        
                        # Extract the answer
                        rag_answer = remaining_text[:next_section_pos].strip()
                        
                        # Clean up: remove any trailing markers or empty lines
                        if rag_answer:
                            rag_answer = "\n".join([line for line in rag_answer.split("\n") if line.strip()]).strip()
                
                # If extraction failed but we found "Rag_Agent > ", try a simpler approach
                if (not rag_answer or len(rag_answer) < 5) and isinstance(rag_response, str):
                    if "Rag_Agent > " in rag_response:
                        # Fallback: get everything after the last "Rag_Agent > " until end or next marker
                        parts = rag_response.rsplit("Rag_Agent > ", 1)
                        if len(parts) > 1:
                            potential_answer = parts[1].strip()
                            # Remove trailing session markers (try various patterns)
                            markers_to_remove = [
                                "\n ### Continue session",
                                "\n### Continue session",
                                "\n ###",
                                "\n###",
                                "\nUser >",
                                "\nUser>",
                                "\n Continue session",
                                "\nContinue session"
                            ]
                            for marker in markers_to_remove:
                                if marker in potential_answer:
                                    potential_answer = potential_answer.split(marker)[0].strip()
                                    break
                            if potential_answer:
                                rag_answer = potential_answer
                
                # Accept answers that are at least 5 characters (reduced from 10 to handle short responses)
                if rag_answer and len(rag_answer) >= 5:
                    # Include web search information if available
                    web_search_info = rag_agent.web_search_calls.copy() if rag_agent.web_search_calls else None
                    result_dict[item_id] = (rag_answer, None, web_search_info)
                else:
                    # Debug: print what we found to help diagnose
                    if isinstance(rag_response, str):
                        has_rag_agent = "Rag_Agent" in rag_response
                        if has_rag_agent:
                            # Print a snippet of the response to see what's happening
                            rag_pos = rag_response.rfind("Rag_Agent")
                            snippet = rag_response[max(0, rag_pos-50):min(len(rag_response), rag_pos+200)]
                            print(f"DEBUG item {item_id}: Found Rag_Agent at pos {rag_pos}")
                            print(f"DEBUG item {item_id}: Response snippet: {repr(snippet)}")
                            print(f"DEBUG item {item_id}: rag_answer={repr(rag_answer)}, length={len(rag_answer) if rag_answer else 0}")
                    result_dict[item_id] = (None, "No RAG answer found in response", None)
            except Exception as e:
                result_dict[item_id] = (None, str(e))
    except Exception as e:
        # If initialization fails, mark all pending requests with error
        print(f"RAG worker initialization failed: {e}")
        while True:
            try:
                request = rag_queue.get_nowait()
                if request is None:
                    break
                item_id, _ = request
                result_dict[item_id] = (None, f"RAG worker initialization failed: {str(e)}")
            except:
                break

class Generate:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o", openai_api_base="", num_processes=None, 
    embed_model_name="BAAI/bge-base-en-v1.5", 
    test_model="Qwen2.5-VL-3B-Instruct",
    device="None"):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.offline_model = model_name
        self.model_name = model_name.split("/")[-1]
        self.openai_api_base = openai_api_base
        # If the number of processes is not specified, use the number of CPU cores
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        # Store RAG config instead of initializing here (to avoid GPU memory issues)
        self.test_model = test_model
        self.embed_model_name = embed_model_name
        self.device = device

    def get_prompt(self, item):
        question = item["question"]
        user_prompt = f"{question}"
        images = item.get("images", [])
        new_images = []
        for i in range(len(images)):
            dir_path = os.path.dirname(os.path.abspath(self.raw_data_file))  
            new_path = dir_path + "/" + images[i]
            if not os.path.exists(new_path):
                print(f"Image path {new_path} does not exist. Please check the input data.")
                continue
            new_images.append(new_path)
        return {"user": user_prompt, "images": new_images}

    # Function to handle item processing
    def process_item(self, args):
        item, model_name, output_file, lock, rag_queue, rag_result_dict = args
        prompt = self.get_prompt(item)
        response = None
        last_exception = None
        self.max_retries = 5
        self.retry_delay = 5  # seconds
        item_id = item.get('id', 'unknown')
        
        # Request RAG response via queue
        enhanced_query = prompt["user"]  # Default to original query
        rag_implementation = False  # Will be True if RAG succeeds
        rag_status = None  # Will contain "successful" or error message
        if rag_queue is not None:
            rag_queue.put((item_id, prompt["user"]))
            # Wait for RAG response (with timeout)
            max_wait_time = 60  # seconds
            wait_interval = 0.1  # seconds
            waited = 0
            while item_id not in rag_result_dict and waited < max_wait_time:
                time.sleep(wait_interval)
                waited += wait_interval
            
            if item_id in rag_result_dict:
                rag_result = rag_result_dict[item_id]
                # Handle both old format (rag_answer, rag_error) and new format (rag_answer, rag_error, web_search_info)
                if len(rag_result) == 3:
                    rag_answer, rag_error, web_search_info = rag_result
                else:
                    rag_answer, rag_error = rag_result
                    web_search_info = None
                
                if rag_answer and rag_error is None:
                    enhanced_query = f"{prompt['user']}\n\nadditional context: {rag_answer}"
                    rag_implementation = True
                    rag_status = "successful"
                    # Store web search information in the item
                    if web_search_info:
                        item["RAG_web_search"] = web_search_info
                        item["RAG_web_search_performed"] = True
                    else:
                        item["RAG_web_search_performed"] = False
                else:
                    print(f"RAG agent failed for item {item_id}: {rag_error}. Using original query.")
                    rag_implementation = False
                    rag_status = rag_error if rag_error else "No RAG answer found in response"
                    item["RAG_web_search_performed"] = False
            else:
                print(f"RAG response timeout for item {item_id}. Using original query.")
                rag_implementation = False
                rag_status = "timeout"
                item["RAG_web_search_performed"] = False
        else:
            # RAG queue is None (RAG disabled)
            rag_implementation = False
            rag_status = "rag_disabled"
            item["RAG_web_search_performed"] = False
        
        if not rag_implementation:
            item["RAG_implementation"] = False
            item["RAG_status"] = rag_status
            # Clean up RAG result from shared dict if it exists
            if rag_result_dict is not None and item_id in rag_result_dict:
                del rag_result_dict[item_id]
            # Write item without model response and return early
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            return item.get('id')
        else:
            for attempt in range(self.max_retries):
                try:
                    # Initialize the client based on the model name
                    if self.model_name.startswith("gpt"):
                        client = OpenAI_Chat(model_name=model_name, messages=[])
                    else:
                        client = Client(model_name=self.offline_model, openai_api_base=self.openai_api_base, messages=[])
                    
                    response = client.chat(prompt=enhanced_query, images=prompt["images"])
                    item[model_name] = response
                    # item["info"] = client.info() # Uncomment if needed
                    item["history"] = client.get_history()
                    break # Exit retry loop on success

                except Exception as e:
                    last_exception = e # Store the exception
                    print(f"Attempt {attempt + 1}/{self.max_retries} failed for item {item_id}: {e}")
                    if attempt < self.max_retries - 1:
                        
                        print(f"Waiting {self.retry_delay} seconds before retrying...")
                        time.sleep(self.retry_delay)
                    else:
                        # Max retries reached
                        print(f"Max retries ({self.max_retries}) reached for item {item_id}. Marking as failed.")
                        item[model_name] = -1 # Mark as failed after all retries
            
            # Clean up RAG result from shared dict
            if rag_result_dict is not None and item_id in rag_result_dict:
                del rag_result_dict[item_id]
            
            item["RAG_implementation"] = True
            item["RAG_status"] = "successful"
 
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return item.get('id')

    def generate(self):
        # Read the raw data file
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        # Check if the output file exists and read processed items
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if self.model_name in item and item[self.model_name] != -1 and item[self.model_name] != None:
                            processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        # Handle potentially corrupt JSON lines
                        continue
                    
        total_items = len(data)
        already_processed = len(processed_ids)
        items_to_process = [item for item in data if item.get('id') not in processed_ids]

        if already_processed > 0:
            print(f"Resuming processing: {already_processed} items already completed, {len(items_to_process)} items remaining.")
        else:
            print(f"Starting fresh: Processing {len(items_to_process)} items.")
        
        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            rag_queue = manager.Queue()
            rag_result_dict = manager.dict()  # Shared dict to store RAG results
            
            # Start RAG worker process (single instance to save GPU memory)
            rag_process = multiprocessing.Process(
                target=rag_worker_process,
                args=(rag_queue, rag_result_dict, self.test_model, self.embed_model_name, self.device, self.openai_api_base)
            )
            rag_process.start()
            
            # Initialize the process pool with the specified number of processes
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(item, self.model_name, self.output_file, lock, rag_queue, rag_result_dict) for item in items_to_process]
            
            # Use tqdm to show progress
            try:
                for _ in tqdm(pool.imap_unordered(self.process_item, args_list), total=len(args_list), desc="Processing items"):
                    pass
            finally:
                # Signal RAG worker to stop
                rag_queue.put(None)
                pool.close()
                pool.join()
                rag_process.join(timeout=30)  # Wait for RAG worker to finish
                if rag_process.is_alive():
                    print("RAG worker did not terminate gracefully, forcing termination...")
                    rag_process.terminate()
                    rag_process.join()
        
        print("Processing completed.")
        print(f"Summary: {len(processed_ids)} items processed, {len(items_to_process)} items remaining.")
        self.cleanup_output(len(data))

    def cleanup_output(self, data_length):
        valid_items = []
        
        with open(self.output_file, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if self.model_name in item and item[self.model_name] != -1 and item[self.model_name] != None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--openai_api_base", type=str, default="", help="Base URL for OpenAI API.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--embed_model_name", type=str, default="BAAI/bge-base-en-v1.5", help="Embedding model name to use.")
    parser.add_argument("--test_model", type=str, default="Qwen2.5-VL-3B-Instruct", help="Test model name to use.")
    parser.add_argument("--device", type=str, default="None", help="Device to use.")
    args = parser.parse_args()

    reformatter = Generate(raw_data_file=args.input_file, output_file=args.output_file, model_name=args.model_name, num_processes=args.num_processes, openai_api_base=args.openai_api_base, embed_model_name=args.embed_model_name, test_model=args.test_model, device=args.device)
    reformatter.generate()
