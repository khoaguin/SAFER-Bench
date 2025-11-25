"""fedrag: A Flower Federated RAG app."""

import os
import re

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid deadlocks during tokenization


class LLMQuerier:
    def __init__(self, model_name, use_gpu=False, gguf_file=None):
        # Determine device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading LLM model: {model_name}")

        # Check if this is a GGUF model
        is_gguf = gguf_file is not None or "GGUF" in model_name

        if is_gguf:
            self._load_gguf_model(model_name, gguf_file, use_gpu)
            self.model_type = "gguf"
            self.tokenizer = None
        else:
            self._load_transformers_model(model_name)
            self.model_type = "transformers"

    def _load_gguf_model(self, model_name, gguf_file, use_gpu):
        """Load GGUF quantized model with llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )
            raise

        logger.info(f"📦 Loading GGUF model: {model_name}")
        if gguf_file:
            logger.info(f"📄 GGUF file: {gguf_file}")

        # Determine GPU layers (-1 = use all layers on GPU)
        n_gpu_layers = -1 if use_gpu or torch.backends.mps.is_available() else 0

        logger.info(
            "⬇️  Downloading model from HuggingFace Hub (first run: ~15-20 min for 70B models)..."
        )
        logger.info("💾 Cache location: ~/.cache/huggingface/hub/")

        # Load GGUF model from HuggingFace Hub
        self.model = Llama.from_pretrained(
            repo_id=model_name,
            filename=gguf_file,
            n_ctx=8192,
            n_gpu_layers=n_gpu_layers,
            verbose=True,
        )

        logger.info("✅ Successfully loaded GGUF model with Metal acceleration")
        logger.info(f"🚀 GPU layers: {n_gpu_layers} (-1 = all layers on GPU)")

    def _load_transformers_model(self, model_name):
        """Load standard Transformers model."""
        logger.info(f"📦 Loading Transformers model: {model_name}")

        # Load model with appropriate settings for size
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        logger.info(f"✅ Successfully loaded model: {self.model.config._name_or_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if empty
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )

    def answer(self, question, documents, options, dataset_name, max_new_tokens):
        """Generate answer for the given question using retrieved documents."""
        # Format options as A) ... B) ... etc.
        formatted_options = "\n".join([f"{k}) {v}" for k, v in options.items()])

        prompt = self.__format_prompt(
            question, documents, formatted_options, dataset_name
        )

        if self.model_type == "gguf":
            return self._answer_gguf(prompt, max_new_tokens)
        else:
            return self._answer_transformers(prompt, max_new_tokens)

    def _answer_gguf(self, prompt, max_new_tokens):
        """Generate answer using GGUF model."""
        logger.info(
            f"🔮 Generating output with GGUF model, max_tokens: {max_new_tokens}"
        )

        response = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.7,
            stop=["</s>", "\n\n"],
            echo=False,
        )

        generated_text = prompt + response["choices"][0]["text"]
        generated_answer = self.__extract_answer(generated_text, prompt)

        logger.info(f"📝 Generated: {response['choices'][0]['text'].strip()[:200]}...")
        logger.info(f"✓ Extracted answer: {generated_answer}")

        return prompt, generated_answer

    def _answer_transformers(self, prompt, max_new_tokens):
        """Generate answer using Transformers model."""
        inputs = self.tokenizer(
            prompt, padding=True, return_tensors="pt", truncation=True
        ).to(self.device)

        attention_mask = (inputs.input_ids != self.tokenizer.pad_token_id).long()

        logger.info(
            f"🔮 Generating output with Transformers, max_tokens: {max_new_tokens}"
        )

        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            early_stopping=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = self.__extract_answer(generated_text, prompt)

        logger.info(f"📝 Generated: {generated_text[len(prompt):].strip()[:200]}...")
        logger.info(f"✓ Extracted answer: {generated_answer}")

        return prompt, generated_answer

    @classmethod
    def __format_prompt(cls, question, documents, options, dataset_name):
        """Format the prompt with instructions, documents, and options."""
        instruction = "You are a helpful medical expert, and your task is to answer a medical question using the relevant documents."

        if dataset_name == "pubmedqa":
            instruction = "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe."
        elif dataset_name == "bioasq":
            instruction = "You are an advanced biomedical AI assistant trained to understand and process medical and scientific texts. Given a biomedical question, your goal is to provide a concise and accurate answer based on relevant scientific literature."

        ctx_documents = "\n".join(
            [f"Document {i + 1}: {doc}" for i, doc in enumerate(documents)]
        )

        prompt = f"""{instruction}

            Here are the relevant documents:
            {ctx_documents}

            Question:
            {question}

            Options:
            {options}

            Answer only with the correct option: """
        return prompt

    @classmethod
    def __extract_answer(cls, generated_text, original_prompt):
        """Extract the answer option (A, B, C, or D) from generated text."""
        # Extract only the new generated text
        response = generated_text[len(original_prompt) :].strip()

        # First try to find letter options A-D (case-insensitive)
        option = re.search(r"\b([A-Da-d])\b", response)
        if option:
            return option.group(1).upper()

        # If not found, try to find number options 1-4 and map to A-D
        number_option = re.search(r"\b([1-4])\b", response)
        if number_option:
            number_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
            return number_to_letter[number_option.group(1)]

        return None
