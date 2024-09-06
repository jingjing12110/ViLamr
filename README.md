<div align="center">
<h1>:hear_no_evil: ViLamr :hear_no_evil:</h1> 
<h2>MLLMs towards Chain-of-Thought Reasoning</h2> 

[//]: # (> [[Paper]&#40;&#41;] [[📝 Project Page]&#40;https://mm-vl.github.io/vilamr/&#41;] )
> [[🤗 Data](https://huggingface.co/datasets/JingjingJiang/MCoT-Instruct-266K)] [[🤗 Model Zoo](https://huggingface.co/collections/JingjingJiang/vilamr-66d02b6b74809ac0b6b09675)]
</div>


## 📢 Latest Updates

- [2024/09/06] 📌 First release of ViLamr-Llama3-8B.
- [2024/09/06] 📌 First release of ViLamr-Vicuna-13B.

## Model Zoo

[//]: # (ViLamr can be instantiated with popular LLMs &#40;e.g., vicuna-13b-v1.5, Llama3.1-8b&#41;.)

> [!NOTE]
> We provide the following trained weights of **ViLamr <mark>w/o RPE</mark>** for versatility:

| ViLamr Weights                                                                      | LLM                  | Connector                                                                         | VE                                        | MMStar |   MMMU-v | RWQA |   MMB-d | GD-VCR | GeoQA | WeMath | M3CoT (ZS) |
|:------------------------------------------------------------------------------------| :------------------- |:-------------------------------------------------------------------------------------|:------------------------------------------|-------:|---------:|-----:|--------:|-------:|------:|-------:|-----------:|
| ViLamr-Vicuna-13B [[🤗 HF](https://huggingface.co/JingjingJiang/vilamr-vicuna-13b)] | Vicuna-13b-v1.5      | GateMixer [[🤗 HF](https://huggingface.co/JingjingJiang/vilamr-vicuna-13b-pretrain)] | <li> CLIP-ViT-L/14 <li> CLIP-ConvNeXt-XXL |   43.0 |     42.3 | 62.2 |    76.0 |   87.9 |  64.8 |   32.2 |       45.2 |
| ViLamr-Llama3-8B [[🤗 HF](https://huggingface.co/JingjingJiang/vilamr-llama3-8b)]   | Llama3.1-8B-Instruct | GateMixer [[🤗 HF](https://huggingface.co/JingjingJiang/vilamr-llama3-8b-pretrain)]  | <li> CLIP-ViT-L/14 <li> CLIP-ConvNeXt-XXL  |   43.5 |     41.4 | 60.1 |    75.3 |   87.0 |  70.8 |   33.0 |       44.0 |

---

## Install

```bash
git https://github.com/jingjing12110/ViLamr.git
cd ViLamr
conda create -n vilamr python=3.10 -y
conda activate vilamr

pip install --upgrade pip   
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
# pip install flash-attn --no-build-isolation --use-pep517
```

## Train

ViLamr is trained using a two-stage strategy on 8 A800 GPUs with 80GB memory.

### Downloading pretrained weights

- [Vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)
- [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [CLIP-ViT-L/14-336px](https://huggingface.co/openai/clip-vit-large-patch14-336)
- [CLIP-ConvNeXt-XXLarge](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup)

### Stage I: Vision-Language Alignment Pretraining

We use a subset of ShareGPT4V with 676K high-quality image-text pairs to the vision-language connector, GateMixer. You can follow [ShareGPT4V's guidance](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md) to prepare pretraining data. The filtered image-text pairs can be obtained from [sharegpt4v_filtered_676k.json](https://huggingface.co/datasets/JingjingJiang/ViLamr-Pretrain).

> [!TIP]
> You are recommended to download the connector weights from [here](https://huggingface.co/collections/JingjingJiang/vilamr-66d02b6b74809ac0b6b09675) directly.

- Training ViLamr-Llama3-8B's connector

```bash
scripts/vilamr_llama3_8b/pretrain.sh
```

- Training ViLamr-Vicuna-13B's connector

```bash
scripts/vilamr_vicuna_13b/pretrain.sh
```

### Stage II: Multimodal Chain-of-Thought Tuning

- Preparing data

> Please download the multimodal instruction-following dataset, [mcot_instruct_v1_266k.json](https://huggingface.co/datasets/JingjingJiang/MCoT-Instruct-266K), and download the images from constituting datasets:
>
> - coco: [train2017](http://images.cocodataset.org/zips/train2017.zip), [val2017](http://images.cocodataset.org/zips/val2017.zip)
> - geo170k: [images](https://huggingface.co/datasets/Luckyjhg/Geo170K/blob/main/images.zip)
> - scienceqa: [train&val](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHevg)
> - tabmwp_mc: [tables](https://github.com/lupantech/PromptPG/tree/main/data/tabmwp)
> - vcr1images: [images](https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip)

> After downloading all of them, organize the data as follows in `./playground/data`,

```
├── mcot_instruct_v1_266k.json
├── images
│   ├── coco
│   │   ├── train2017
│   │   └── val2017
│   ├── geo170k
│   │   ├── geo3k
│   │   └── geoqa_plus
│   ├── scienceqa
│   │   ├── train
│   │   └── val
│   ├── tabmwp_mc
│   ├── vcr1images
```

- Training ViLamr-Llama3-8B

```bash
scripts/vilamr_llama3_8b/finetune_base.sh
```

- Training ViLamr-Vicuna-13B

```bash
scripts/vilamr_vicuna_13b/finetune_base.sh
```

## Evaluation

- Downloading evaluation benchmarks & dataset

> - [MMStar](https://mmstar-benchmark.github.io/)
> - [MMMU val](https://mmmu-benchmark.github.io/)
> - [RWQA](https://x.ai/blog/grok-1.5v)
> - [MMB-dev](https://github.com/open-compass/MMBench)
> - [GD-VCR](https://gd-vcr.github.io/)
> - [GeoQA](https://github.com/pipilurj/G-LLaVA)
> - [WeMath](https://we-math.github.io/)
> - [M3CoT](https://lightchen233.github.io/m3cot.github.io/index.html)

- Multi-GPU inference

```bash
scripts/vilamr_vicuna_13b/inference_base.sh
scripts/vilamr_llama3_8b/inference_base.sh
```

- Accuracy calculation

```bash
scripts/acc_calculate.sh
```

## Acknowledgement

[Vicuna](https://github.com/lm-sys/FastChat) | [Llama](https://github.com/lm-sys/FastChat) | [LLaVA](https://github.com/haotian-liu/LLaVA) | [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) | [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V) | [LLaVA-MORE](https://github.com/aimagelab/LLaVA-MORE)

[//]: #
[//]: #

