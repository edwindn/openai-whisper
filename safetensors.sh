wget https://huggingface.co/openai/whisper-large-v3/resolve/main/pytorch_model.fp32-00001-of-00002.bin?download=true
wget https://huggingface.co/openai/whisper-large-v3/resolve/main/pytorch_model.fp32-00002-of-00002.bin?download=true
mv pytorch_model.fp32-00001-of-00002.bin?download=true whisper_01.bin && mv pytorch_model.fp32-00002-of-00002.bin?download=true whisper_02.bin
wget https://huggingface.co/openai/whisper-large-v3/resolve/main/tokenizer.json?download=true && mv 'tokenizer.json?download=true' tokenizer.json
