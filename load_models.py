from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AIModelManager:
    _intent_classifier = None  
    _tokenizer = None
    device = "cpu"

    @staticmethod
    def load_models():
        """ 📌 **AI modellerini yükler ve sadece bir kez başlatır.** """
        if AIModelManager._intent_classifier is None:
            model_id = "google/gemma-2-2b-it"

            # **Cihazı belirle (MPS, CUDA veya CPU)**
            if torch.backends.mps.is_available():
                AIModelManager.device = "mps"
            elif torch.cuda.is_available():
                AIModelManager.device = "cuda"
            else:
                AIModelManager.device = "cpu"

            print(f"🚀 **AI modeli yükleniyor... (Device: {AIModelManager.device})**")

            AIModelManager._tokenizer = AutoTokenizer.from_pretrained(model_id)
            AIModelManager._tokenizer.pad_token = AIModelManager._tokenizer.eos_token  

            AIModelManager._intent_classifier = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if AIModelManager.device != "cpu" else torch.float32,
                device_map=AIModelManager.device
            )

            print(f"✅ **AI modeli başarıyla yüklendi! (Device: {AIModelManager.device})**")

    @staticmethod
    def get_intent_classifier():
        """ 📌 Yüklü modeli döndürür, yoksa yükler. """
        if AIModelManager._intent_classifier is None:
            AIModelManager.load_models()
        return AIModelManager._intent_classifier, AIModelManager._tokenizer

    @staticmethod
    def get_device():
        """ 📌 Kullanılan cihazı döndürür. """
        return AIModelManager.device


if __name__ == "__main__":
    AIModelManager.load_models()
