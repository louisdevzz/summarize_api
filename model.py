from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from datasets import Dataset
import numpy as np
import pickle
import os
import pandas as pd

class VietnameseSummarizer:
    def __init__(self, model_name="pengold/t5-vietnamese-summarization", max_length=512):
        # Add load_from_path parameter to allow loading from saved model
        if model_name.startswith('./') or model_name.startswith('/'): 
            # Load from local path
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        else:
            # Load from HuggingFace hub
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def prepare_dataset(self, texts, summaries):
        """Prepare dataset for training"""
        dataset_dict = {
            "text": texts,
            "summary": summaries
        }
        return Dataset.from_dict(dataset_dict)

    def preprocess_function(self, examples):
        """Preprocess the data for training"""
        inputs = self.tokenizer(
            examples["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        targets = self.tokenizer(
            examples["summary"],
            max_length=150,
            truncation=True,
            padding="max_length"
        )

        inputs["labels"] = targets["input_ids"]
        return inputs

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        return {"decoded_preds": decoded_preds, "decoded_labels": decoded_labels}

    def train(self, train_texts, train_summaries, 
              output_dir="./vietnamese-summarizer-finetuned",
              num_train_epochs=3,
              per_device_train_batch_size=8,
              save_steps=1000):
        """Fine-tune the model on custom data"""
        
        # Prepare training dataset
        train_dataset = self.prepare_dataset(train_texts, train_summaries)
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            save_steps=save_steps,
            save_total_limit=2,
            predict_with_generate=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        trainer.train()

    def train_model(self,path_save="./saved_models/vietnamese_summarizer"):
        # Assuming your CSV has columns named 'text' and 'summary'
        train_texts, train_summaries = VietnameseSummarizer.load_data_from_csv(
            csv_path="vietnamese_articles.csv",  # Updated to use the actual CSV file
            text_column="text",
            summary_column="summary",
            encoding='utf-8'
        )
        
        # 3. Train the model with increased epochs
        self.train(
            train_texts=train_texts,
            train_summaries=train_summaries,
            output_dir="./training_checkpoint",
            num_train_epochs=3,
        )
        
        # 4. Save the model
        self.save_model(path_save)
        print("Model saved successfully")

    def summarize(self, text, max_summary_length=150, min_summary_length=30):
        # Prepare the input text
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )

        # Decode the generated summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary

    def save_model(self, path):
        """
        Save the model to a PKL file
        Args:
            path: Path to save the model (without extension)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Move model to CPU before saving
        self.model = self.model.cpu()
        
        # Save model state
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'max_length': self.max_length
        }

        full_path = f"{path}.pkl"
        torch.save(model_state, full_path)

        # Also save the tokenizer separately
        tokenizer_path = os.path.join(os.path.dirname(path), 'tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Move model back to original device
        self.model = self.model.to(self.device)
        
        print(f"Model saved to {full_path}")
        print(f"Tokenizer saved to {tokenizer_path}")

    @classmethod
    def load_model(cls, path):
        """
        Load the model from a PKL file
        Args:
            path: Path to load the model (without extension)
        """
        # Initialize with base model
        instance = cls()
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        full_path = f"{path}.pkl"
        # Load the model state with explicit CPU mapping
        model_state = torch.load(full_path, map_location=lambda storage, loc: storage)

        # Load model state
        instance.model.load_state_dict(model_state['model_state_dict'])
        instance.max_length = model_state['max_length']

        # Load tokenizer from separate directory
        tokenizer_path = os.path.join(os.path.dirname(path), 'tokenizer')
        instance.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Move model to appropriate device
        instance.model = instance.model.to(device)
        
        return instance

    @staticmethod
    def load_data_from_csv(csv_path, text_column="text", summary_column="summary", encoding='utf-8'):
        """
        Load training data from a CSV file
        Args:
            csv_path: Path to the CSV file
            text_column: Name of the column containing the text to summarize
            summary_column: Name of the column containing the summaries
            encoding: File encoding (default: utf-8)
        Returns:
            texts: List of texts
            summaries: List of summaries
        """
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            
            # Verify columns exist
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV file")
            if summary_column not in df.columns:
                raise ValueError(f"Column '{summary_column}' not found in CSV file")
            
            # Remove rows with missing values
            df = df.dropna(subset=[text_column, summary_column])
            
            # Convert to lists
            texts = df[text_column].tolist()
            summaries = df[summary_column].tolist()
            
            print(f"Loaded {len(texts)} examples from {csv_path}")
            return texts, summaries
            
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            raise

