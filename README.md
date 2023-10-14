# Genius Chatbot
*Version: 0.17.0*

Chatbot that uses any desired hugging face model or allows for scalable 
intelligence based on hardware limitations

### Usage:

| Short Flag | Long Flag            | Description                                                                     |
|------------|----------------------|---------------------------------------------------------------------------------|
| -h         | --help               | See Usage                                                                       |
| -a         | --assimilate         | Assimilate knowledge from media provided in directory                           |
|            | --batch-token        | Number of tokens per batch                                                      |
|            | --chromadb-directory | Number of chunks to use                                                         |
|            | --chunks             | Number of chunks to use                                                         |
| -e         | --embeddings-model   | [Embeddings model](https://www.sbert.net/docs/pretrained_models.html) to use    |
|            | --hide-source        | Hide source of answer                                                           |
| -j         | --json               | Export to JSON                                                                  |
|            | --openai-token       | OpenAI Token                                                                    |
|            | --openai-api         | OpenAI API Url                                                                  |
|            | --pgvector-user      | PGVector user                                                                   |
|            | --pgvector-password  | PGVector password                                                               |
|            | --pgvector-host      | PGVector host                                                                   |
|            | --pgvector-port      | PGVector port                                                                   |
|            | --pgvector-database  | PGVector database                                                               |
|            | --pgvector-driver    | PGVector driver                                                                 |
| -p         | --prompt             | Prompt for chatbot                                                              |
|            | --mute-stream        | Mute stream of generation                                                       |
| -m         | --model              | Copy [GPT4All](https://gpt4all.io/index.html) .bin file from the Model Explorer |
|            | --max-token-limit    | Maximum token to generate                                                       |
|            | --model-directory    | Directory to store models locally                                               |
|            | --model-engine       | GPT4All LlamaCPP, or OpenAI                                                     |

### Example:
Your very first time running should assimilate up to 3 documents to establish the Chroma Database. This will unlock prompting

```bash
genius-chatbot --assimilate "/directory/of/documents"
```

```bash
genius-chatbot --prompt "What is the 10th digit of Pi?"
```

```bash
genius-chatbot --prompt "Chatbots are cool because they" \
    --model "wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_0.bin" \
    --model-engine "GPT4All" \
    --assimilate "/directory/of/documents" \
    --json
```

#### Install Instructions
Install Python Package

Windows Prerequisites:

Visual Studio Code 2022

```bash
winget install -e --id Kitware.CMake
```

Ubuntu Prerequisites:
```bash
apt install -y pandoc
```

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
