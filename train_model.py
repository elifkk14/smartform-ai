import os
import logging
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
)
from transformers import DataCollatorWithPadding

# 📌 Log Ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 📌 MPS Bellek Yönetimi
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

class FormDataManager:
    def __init__(self, final_data_path, enhanced_data_path):
        self.final_data_path = final_data_path
        self.enhanced_data_path = enhanced_data_path

    def load_and_prepare(self):
        """
        🔹 **Veri setlerini yükler, temizler ve model için hazır hale getirir.**
        """
        try:
            logger.info("📂 Veri setleri yükleniyor...")
            final_df = pd.read_csv(self.final_data_path, low_memory=False)
            enhanced_df = pd.read_parquet(self.enhanced_data_path)

            # 🔹 **Etiketleme ve birleştirme**
            final_df['label'] = final_df['target'].astype(str)
            enhanced_df['label'] = enhanced_df['question_type'].astype(str)
            combined_df = pd.concat([final_df, enhanced_df], ignore_index=True)

            # 🔹 **Tek örnek içeren sınıfları kaldır**
            label_counts = combined_df['label'].value_counts()
            if label_counts.min() < 2:
                logger.warning("⚠️ Tek örnek içeren sınıflar kaldırılıyor...")
                valid_labels = label_counts[label_counts > 1].index
                combined_df = combined_df[combined_df['label'].isin(valid_labels)]
                logger.info(f"📊 Güncellenmiş sınıf dağılımı: {combined_df['label'].value_counts()}")

            # 🔹 **NaN değerleri temizle**
            combined_df = combined_df.fillna("unknown")
            logger.info(f"🛠️ NaN içeren sütunlar temizlendi.")

            # 🔹 **Soruları analiz et ve kısa olanları temizle**
            if "question_text" in combined_df.columns:
                combined_df = combined_df[combined_df["question_text"].str.len() > 5]
                logger.info(f"🛠️ 'question_text' filtresi uygulandı, yeni veri seti boyutu: {combined_df.shape}")

            # 🔹 **Geçersiz label değerlerini kaldır**
            invalid_labels = ["unknown", "None"]
            combined_df = combined_df[~combined_df["label"].isin(invalid_labels)]
            logger.warning("⚠️ Geçersiz label değerleri kaldırıldı.")

            # 🔹 **Form ID ve diğer sayısal sütunları string olarak kaydet**
            for col in ["form_id", "question_id", "created_utc", "id"]:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].astype(str)

            # 📌 **Veri bölme işlemi (train/val)**
            try:
                if len(combined_df) < 1000:
                    logger.warning("⚠️ Küçük veri seti! `stratify` kaldırılıyor.")
                    train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)
                else:
                    train_df, val_df = train_test_split(combined_df, test_size=0.2, stratify=combined_df["label"], random_state=42)
            except ValueError as e:
                logger.error(f"❌ Veri bölme işlemi başarısız: {e}")
                raise

            # 🔹 **Boş veri seti kontrolü**
            if train_df.empty or val_df.empty:
                logger.error("❌ HATA: Eğitim veya doğrulama veri kümesi boş! Model eğitilemez.")
                raise ValueError("Eğitim veya doğrulama veri kümesi boş!")

            logger.info(f"✅ Eğitim veri kümesi boyutu: {train_df.shape}")
            logger.info(f"✅ Doğrulama veri kümesi boyutu: {val_df.shape}")

            return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)

        except Exception as e:
            logger.error(f"❌ Veri yüklenirken hata oluştu: {e}", exc_info=True)
            raise

class FormTrainer:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=5, device=None):
        """
        🔹 **AI Modeli Tanımlama**
        - `model_name`: Hugging Face model ismi.
        - `num_labels`: Sınıf sayısı.
        """
        self.device = device if device else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

        # **Veri Pad'leme için Collator**
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _tokenize(self, batch):
        return self.tokenizer(
            batch["question_text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    def train(self, train_dataset, val_dataset):
        """
        🔹 **Modeli eğitir ve kaydeder.**
        """
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            logging_dir="./logs",
            learning_rate=3e-5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        try:
            trainer.train()
            self.model.save_pretrained("models/form_classifier")
            logger.info("✅ Model eğitildi ve kaydedildi: models/form_classifier")
        except Exception as e:
            logger.error(f"❌ Model eğitimi başarısız: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        logger.info("🚀 Model eğitimi başlatılıyor...")

        # 📌 **Veriyi Yükleme**
        data_manager = FormDataManager(
            final_data_path="data/processed/final_dataset.csv",
            enhanced_data_path="data/processed/enhanced_dataset.parquet"
        )
        train_dataset, val_dataset = data_manager.load_and_prepare()

        # 📌 **Model Eğitimi**
        trainer = FormTrainer(num_labels=5)
        trainer.train(train_dataset, val_dataset)

        logger.info("🎯 Model eğitimi tamamlandı!")

    except Exception as e:
        logger.error(f"❌ Kritik hata: {str(e)}", exc_info=True)
        raise
