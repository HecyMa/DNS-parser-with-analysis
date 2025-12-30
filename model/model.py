from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import re
from peft import PeftModel


class LLModel:
    def __init__(self, base_model_path: str = "./model/models/Qwen3-0.6B", lora_adapter_path: str = "./model/models/Qwen3-0.6B-finetuned"):
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )

        if not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}\n{% elif message['role'] == 'user' %}{{ message['content'] }}\n{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '' }}{% endif %}"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Загрузка модели
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        # Загрузка LoRA-адаптера
        self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)

        # Объединение весов LoRA
        try:
            self.model = self.model.merge_and_unload()
        except:
            print("Не удалось объединить LoRA веса, продолжаем с адаптером")

        self.model.eval()

        # Определение устройства
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Предзагрузка конфигурации генерации
        self.generation_config = GenerationConfig(
            max_new_tokens=300,
            min_new_tokens=30,
            do_sample=False,
            temperature=0.3,
            top_p=0.85,
            top_k=30,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=1,
            early_stopping=True
        )

    def cleanup_response(self, text):
        """Очистка ответа от нежелательных частей"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        text = re.sub(r'\n\s*\n', '\n', text)

        text = re.sub(r'Вывод[аяю].*?(?=\n|$)', '', text, flags=re.DOTALL)
        text = re.sub(r'В заключение.*?(?=\n|$)', '', text, flags=re.DOTALL)
        text = re.sub(r'Таким образом.*?(?=\n|$)', '', text, flags=re.DOTALL)

        text = re.sub(r'(\d+)\)', r'\1.', text)  # Заменяем 1) на 1.

        lines = text.split('\n')
        cleaned_lines = []
        in_advantages = False
        in_disadvantages = False
        advantage_count = 0
        disadvantage_count = 0

        for line in lines:
            line = line.strip()
            if 'Достоинства:' in line or 'Преимущества:' in line:
                in_advantages = True
                in_disadvantages = False
                advantage_count = 0
                cleaned_lines.append('Достоинства:')
            elif 'Недостатки:' in line:
                in_disadvantages = True
                in_advantages = False
                disadvantage_count = 0
                cleaned_lines.append('\nНедостатки:')
            elif in_advantages and re.match(r'^\d+\.', line):
                advantage_count += 1
                if advantage_count <= 3:
                    cleaned_lines.append(line)
            elif in_disadvantages and re.match(r'^\d+\.', line):
                disadvantage_count += 1
                if disadvantage_count <= 3:
                    cleaned_lines.append(line)
            elif line and not any(x in line for x in ['<', 'Вывод', 'Таким образом', 'В заключение']):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def generate(self, prompt, system_message=None):
        """Генерация ответа с улучшенными параметрами"""
        # Улучшенный системный промпт
        if system_message is None:
            system_message = """
                Ты - технический аналитик. Отвечай ТОЛЬКО в следующем формате:

                Достоинства:
                1. [конкретное достоинство №1]
                2. [конкретное достоинство №2]
                3. [конкретное достоинство №3]
                
                Недостатки:
                1. [конкретный недостаток №1]
                2. [конкретный недостаток №2]
                3. [конкретный недостаток №3]
                
                ПРАВИЛА:
                1. Каждый пункт должен быть конкретным техническим фактом
                2. Не добавляй введение, заключение или комментарии
                3. Не используй скобки (), кроме как для нумерации
                4. Отвечай только на основе фактической информации
                5. Если не знаешь, пиши "Недостаточно информации"
                6. Строго ограничься 3 пунктами в каждом разделе
                7. Не используй теги <think> или подобные
            """

        # Формируем сообщения с учетом особенностей Qwen
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Проанализируй технические характеристики. {prompt}"}
        ]

        try:
            # Применяем шаблон чата
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # Fallback шаблон
            text = f"System: {system_message}\n\nUser: Проанализируй технические характеристики и перечисли достоинства и недостатки: {prompt}\n\nAssistant:"

        # Токенизация
        model_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Генерация с конфигом
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                generation_config=self.generation_config
            )

        # Декодируем только сгенерированную часть
        input_length = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_length:].tolist()

        # Останавливаемся на первом eos_token
        if self.tokenizer.eos_token_id in output_ids:
            eos_index = output_ids.index(self.tokenizer.eos_token_id)
            output_ids = output_ids[:eos_index]

        content = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        # Очистка ответа
        content = self.cleanup_response(content)

        return content


model = LLModel()
