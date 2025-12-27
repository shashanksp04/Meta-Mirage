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
        self.rag_agent = MainAgent(test_model=test_model, embed_model_name=embed_model_name, device=device)
        self.rag_runner = self.rag_agent.main()

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
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)
        response = None
        last_exception = None
        self.max_retries = 5
        self.retry_delay = 5  # seconds
        item_id = item.get('id', 'unknown')
        for attempt in range(self.max_retries):
            try:
                # Initialize the client based on the model name
                if self.model_name.startswith("gpt"):
                    client = OpenAI_Chat(model_name=model_name, messages=[])
                else:
                    client = Client(model_name=self.offline_model, openai_api_base=self.openai_api_base, messages=[])
                
                # Get RAG agent response and enhance query
                enhanced_query = prompt["user"]  # Default to original query
                try:
                    rag_response = self.rag_runner.run_debug(prompt["user"])
                    # Extract agent's answer (everything after "Rag_Agent > ")
                    if "Rag_Agent > " in rag_response:
                        rag_answer = rag_response.split("Rag_Agent > ", 1)[1].strip()
                        enhanced_query = f"{prompt['user']}\n\nadditional context: {rag_answer}"
                except Exception as rag_error:
                    # If RAG fails, use original query
                    print(f"RAG agent failed for item {item_id}: {rag_error}. Using original query.")
                
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
                    
        items_to_process = [item for item in data if item.get('id') not in processed_ids]
        print(f"Processing {len(items_to_process)} items.")
        
        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            # Initialize the process pool with the specified number of processes
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(item, self.model_name, self.output_file, lock) for item in items_to_process]
            # Use tqdm to show progress
            for _ in tqdm(pool.imap_unordered(self.process_item, args_list), total=len(args_list), desc="Processing items"):
                pass
            pool.close()
            pool.join()
        
        print("Processing completed.")
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
