# [Q] J'ai réussi à fine-tuner Qwen2.5-Omni-3B en gardant les capacités multimodales - est-ce que c'est aussi difficile que j'ai l'impression ?

Salut à tous,

Je travaille sur un projet perso (IA pour l'agriculture) et je viens de passer **20+ heures non-stop** à fine-tuner Qwen2.5-Omni-3B. Je voulais avoir votre avis : est-ce que ce que j'ai fait est considéré comme complexe, ou j'ai juste galéré pour rien ?

## Mon objectif

Fine-tuner le modèle sur mes données (17 exemples de conversations spécialisées) **TOUT EN gardant les capacités multimodales intactes** (audio, vision, vidéo). Pas question de perdre le "Omni" juste pour un fine-tuning texte.

## Ce qui a foiré

- **SFTTrainer ne fonctionne pas** avec l'architecture Omni (pas de `forward()` implémenté sur le wrapper principal)
- Le modèle a une archi bizarre : `Qwen2_5OmniForConditionalGeneration` → `thinker` (Thinker) + `talker` (Talker)
- Impossible d'utiliser les approches classiques de fine-tuning
- Des dizaines d'erreurs en cascade :
  - Missing `model.safetensors.index.json`
  - PyTorch CVE-2025-32434 → upgrade vers PyTorch 2.6
  - Missing `preprocessor_config.json`, `chat_template.json`, `tokenizer_config.json`
  - SFTTrainer API changes (`tokenizer` → `processing_class`, etc.)
  - Et surtout : **`_forward_unimplemented()` error**

## Ma solution (après des dizaines de tentatives)

1. J'ai créé un **wrapper custom** autour du modèle Omni
2. J'ai extrait le `thinker` (le vrai modèle génératif)
3. J'ai appliqué **LoRA directement sur le Thinker AVANT de wrapper**
4. Mon wrapper expose juste un `forward()` qui appelle le Thinker
5. **QLoRA (4-bit)** pour tenir dans 7.5GB VRAM (RTX 3080)

### Code simplifié du wrapper

```python
class Qwen2_5OmniWrapper(nn.Module):
    def __init__(self, omni_model):
        super().__init__()
        self.omni_model = omni_model
        self.thinker = omni_model.thinker  # Accès au Thinker
        self.config = omni_model.config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Retirer les clés multimodales
        kwargs_clean = {k: v for k, v in kwargs.items()
                       if k not in ['pixel_values', 'audio_values', 'video_values']}

        # Appeler le Thinker directement
        outputs = self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs_clean
        )
        return outputs

    def generate(self, *args, **kwargs):
        # Déléguer au modèle Omni complet pour génération multimodale
        return self.omni_model.generate(*args, **kwargs)
```

### Le truc crucial découvert après BEAUCOUP d'essais

Il faut **absolument appliquer LoRA sur le Thinker AVANT de créer le wrapper**, sinon les gradients ne se propagent pas :

```python
# 1. Extraire le Thinker
thinker = omni_model.thinker

# 2. Appliquer LoRA sur le Thinker
thinker_with_lora = get_peft_model(thinker, lora_config)

# 3. Remplacer le Thinker dans omni_model
omni_model.thinker = thinker_with_lora

# 4. ENSUITE créer le wrapper
model = Qwen2_5OmniWrapper(omni_model)
```

Si tu appliques LoRA après le wrapper, les gradients vont direct vers le Thinker sans passer par les adapters LoRA. **Erreur : `None of the inputs have requires_grad=True`**

## Résultat

- ✅ Training lancé avec succès
- ✅ Loss qui descend (8.83 → en baisse)
- ✅ Seulement **0.87% des paramètres entraînés** (41M/4.7B)
- ✅ Le modèle garde **toute son architecture multimodale**
- ✅ QLoRA 4-bit : ~7.5GB VRAM utilisé

Configuration :
- Batch size: 1 (gradient accumulation: 4)
- Learning rate: 2e-4
- Max steps: 100
- LoRA rank: 16
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## Ma question

Est-ce que c'est **normal** d'avoir dû bidouiller autant ? Est-ce que quelqu'un a déjà fine-tuné un modèle Omni/multimodal en gardant toutes ses capacités ? Ou est-ce que j'ai juste pris le chemin le plus compliqué ?

Je suis un développeur assez têtu (j'étais prêt à continuer 40h de plus s'il le fallait lol), mais j'aimerais savoir si c'est quelque chose de "normal" dans le domaine ou si j'ai fait quelque chose d'inhabituel.

Merci pour vos retours !

---

## TL;DR

Fine-tuné Qwen2.5-Omni en gardant les capacités multimodales via un **wrapper custom + LoRA sur le Thinker**. 20h de galère. C'est normal ou j'ai overcompliqué ?

---

**Edit:** Pour ceux qui veulent les détails techniques, j'ai tout documenté dans mon repo (je peux partager si ça intéresse).

**Stack technique :**
- Docker + NVIDIA runtime (CUDA 12.3.2)
- PyTorch 2.6.0 + CUDA 12.4
- Transformers (commit `3a1ead0` pour support Qwen2.5-Omni)
- PEFT (LoRA adapters)
- bitsandbytes (4-bit quantization)
- Dataset: 17 exemples JSONL (chat + analysis avec contexte JSON)
