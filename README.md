# Genius Chatbot

![PyPI - Version](https://img.shields.io/pypi/v/genius-chatbot)
![PyPI - Downloads](https://img.shields.io/pypi/dd/genius-chatbot)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/genius-chatbot)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/genius-chatbot)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/genius-chatbot)
![PyPI - License](https://img.shields.io/pypi/l/genius-chatbot)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/genius-chatbot)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/genius-chatbot)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/genius-chatbot)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/genius-chatbot)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/genius-chatbot)

![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/genius-chatbot)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/genius-chatbot)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/genius-chatbot)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/genius-chatbot)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/genius-chatbot)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/genius-chatbot)

*Version: 1.10.1*

Chatbot that uses any hugging face model or OpenAI endpoint. 

Local vector store supported with ChromaDB, or connect to a PGVector database

Allows for scalable intelligence tailored for hardware limitations

This repository is actively maintained - Contributions are welcome!

Contribution Opportunities:
- Support more vector databases
- Get list of models to choose from
- Output everything as JSON when selected

<details>
  <summary><b>Usage:</b></summary>

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

</details>

<details>
  <summary><b>Example:</b></summary>

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

</details>

<details>
  <summary><b>Installation Instructions:</b></summary>

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

</details>

## Geniusbot Application

Use with a GUI through Geniusbot

Visit our [GitHub](https://github.com/Knuckles-Team/geniusbot) for more information

<details>
  <summary><b>Installation Instructions with Geniusbot:</b></summary>

Install Python Package

```bash
python -m pip install geniusbot
```

</details>

<details>
  <summary><b>Repository Owners:</b></summary>


<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)
</details>
