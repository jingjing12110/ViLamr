# MCoT-Instruct

MCoT-Instruct is a multimodal instruction-following dataset with high-quality chain-of-thought (CoT) responses, collected with GPT assistance. It is constructed to improve MLLMs/VLMs towards chain-of-thought reasoning.

| Source dataset                                            | Split           | # of conversation | 
|:----------------------------------------------------------|:----------------|------------------:|
| [A-OKVQA](https://github.com/allenai/aokvqa)              | train+val       |            18,201 |
| [GPT-VQA](https://github.com/opendatalab/MLLM-DataEngine) | round1 & round2 |            26,053 |
| [VCR](https://visualcommonsense.com/)                     | train           |            80,047 |
| [GeoQA-T](https://github.com/pipilurj/G-LLaVA)            | qa_tuning       |           111,353 |
| [ScienceQA-IMG](https://github.com/lupantech/ScienceQA)   | train+val       |             8,315 |
| [TabMWP-MC](https://github.com/lupantech/PromptPG)        | train           |             5,744 |
