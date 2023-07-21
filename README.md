# Genius Chatbot
*Version: 0.12.0*

Chatbot that uses any desired hugging face model or allows for scalable 
intelligence based on hardware limitations

### Usage:

| Short Flag | Long Flag          | Description                                                                  |
|------------|--------------------|------------------------------------------------------------------------------|
| -h         | --help             | See Usage                                                                    |
| -a         | --assimilate       | Assimilate knowledge from media provided in directory                        |
| -b         | --batch-token      | Number of tokens per batch                                                   |
| -c         | --chunks           | Number of chunks to use                                                      |
| -d         | --directory        | Directory for chromadb and model storage                                     |
| -e         | --embeddings-model | [Embeddings model](https://www.sbert.net/docs/pretrained_models.html) to use |
| -j         | --json             | Export to JSON                                                               |
| -p         | --prompt           | Prompt for chatbot                                                           |
| -q         | --mute-stream      | Mute stream of generation                                                    |
| -m         | --model            | Model to use from Huggingface                                                |
| -p         | --prompt           | Prompt for chatbot                                                           |
| -s         | --hide-source      | Hide source of answer                                                        |
| -t         | --max-token-limit  | Maximum token to generate                                                    |
| -x         | --model-engine     | GPT4All or LlamaCPP                                                          |

### Example:
```bash
genius-chatbot --assimilate "/directory/of/documents"
```

```bash
genius-chatbot --prompt "What is the 10th digit of Pi?"
```

```bash
genius-chatbot  \
    --model "wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin" \
    --prompt "Chatbots are cool because they" \
    --model-engine "GPT4All" \
    --assimilate "/directory/of/documents" \
    --json
```

#### Install Instructions
Install Python Package

```bash
python -m pip install genius-chatbot
```

#### Build Instructions
Build Python Package

```bash
sudo chmod +x ./*.py
sudo pip install .
python3 setup.py bdist_wheel --universal
# Test Pypi
twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose -u "Username" -p "Password"
# Prod Pypi
twine upload dist/* --verbose -u "Username" -p "Password"
```
