This directory contains several modified versions of `ppo_trainer.py` from [TRL (Hugging Face)](https://github.com/huggingface/trl). Each version includes two consistent changes:

* The **value head (score layer)** of the value model is **initialized to zero** before training begins.
* The **policy model weights are frozen** — meaning the policy model is **not updated** during training.

### Variants:

* **`init_0_no_scheduler.py`**
  ➤ Uses a **constant learning rate** (no scheduler).

* **`init_0_with_scheduler.py`**
  ➤ Uses **Adam optimizer with learning rate scheduling**.

* **`trainer_v_3.py`**
  ➤ Includes **Adam scheduler**.
  ➤ **Logs predicted values per token** (from the partially trained value model) at various checkpoints.
