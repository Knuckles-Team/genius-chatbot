# Genius Chatbot
*Version: 0.2.0*

Chatbot that uses any desired hugging face model or allows for scalable 
intelligence based on hardware limitations

### Usage:
| Short Flag | Long Flag       | Description                                |
|------------|-----------------|--------------------------------------------|
| -h         | --help          | See Usage                                  |
| -c         | --cuda          | Use Nvidia Cuda instead of CPU             |
| -s         | --save          | Save model locally                         |
| -d         | --directory     | Directory for model                        |
| -o         | --output-length | Maximum output length of response          |
| -p         | --prompt        | Prompt for chatbot                         |
| -m         | --model         | Model to use from Huggingface              |

### Example:
```bash
genius-chatbot --model "facebook/opt-66b" --output-length "500" --prompt "Chatbots are cool because they"
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
