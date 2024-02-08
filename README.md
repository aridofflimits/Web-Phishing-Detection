# Web Phishing Detection with Deep Learning
The main goal of this repository is to identify and classify phishing websites using deep learning techniques. Phishing websites are malicious sites that try to trick users into providing sensitive information, such as usernames, passwords, and credit card details, by pretending to be trustworthy entities. The raw data being analyzed here are URLs and HTML content of websites. URL analysis involves examining the structure and contents of the URL for signs of phishing, such as misspelled domain names or suspicious subdomains. HTML content analysis involves examining the actual contents of the webpage, such as forms that ask for sensitive information, the presence of suspicious scripts, or the overall structure and design of the page.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)

# Introduction
Phishing attacks are increasingly prevalent in today's digital age. These malicious acts aim to deceive users into revealing sensitive information such as usernames, passwords, and credit card details by posing as trustworthy entities. The repercussions of these attacks can be serious, leading to identity theft, financial loss, and significant harm to an individual's or an organization’s reputation. 

Traditionally, the detection of phishing has heavily relied on analyzing URLs. These systems scrutinize suspicious patterns in URLs, such as misspelled domain names or unusual subdomains. However, this approach is not without its flaws. Skilled phishers have devised numerous tactics to evade URL-based detection. For instance, they might employ methods like URL redirection, where the URL initially appears legitimate but redirects the user to a malicious site. Other tactics include using URL shorteners or concealing the malicious URL within seemingly harmless hyperlinks.

In response to this, I have developed a more robust approach to phishing detection in this repository. This approach analyzes not just URLs, but also the HTML content of web pages. By utilizing deep learning techniques, the model can extract and learn from complex features and patterns within the raw data. Deep learning, a branch of machine learning that uses multi-layered neural networks, is particularly effective for this task. It enables the model to delve into the actual contents of web pages, such as forms that ask for sensitive information, the presence of suspicious scripts, or the overall structure and design of the page, going beyond merely examining the structure of URLs.

After processing both the URLs and HTML content, the model predicts whether the webpage is legitimate or a potential phishing threat. By combining URL and HTML content analysis, this approach offers a more comprehensive and reliable phishing detection system. Through this project, I hope to contribute to making the internet a safer place for users.
