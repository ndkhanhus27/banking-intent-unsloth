import os
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
import yaml



# ==============================
# 1. CẤU HÌNH
# ==============================
with open("configs/train.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model_name"]
MAX_SEQ_LENGTH = int(config["max_seq_length"])

BATCH_SIZE = int(config["per_device_train_batch_size"])
GRAD_ACCUM = int(config["gradient_accumulation_steps"])

LEARNING_RATE = float(config["learning_rate"])
EPOCHS = int(config["num_train_epochs"])

LOGGING_STEPS = int(config["logging_steps"])

OUTPUT_DIR = config["output_dir"]
TRAIN_FILE = config["train_file"]

# ==============================
# 2. LABEL MAPPING
# ==============================
id2label = {
    0: "activate_my_card",
    1: "age_limit",
    2: "apple_pay_or_google_pay",
    3: "atm_support",
    4: "automatic_top_up",
    5: "balance_not_updated_after_bank_transfer",
    6: "balance_not_updated_after_cheque_or_cash_deposit",
    7: "beneficiary_not_allowed",
    8: "cancel_transfer",
    9: "card_about_to_expire",
    10: "card_acceptance",
    11: "card_arrival",
    12: "card_delivery_estimate",
    13: "card_linking",
    14: "card_not_working",
    15: "card_payment_fee_charged",
    16: "card_payment_not_recognised",
    17: "card_payment_wrong_exchange_rate",
    18: "card_swallowed",
    19: "cash_withdrawal_charge",
    20: "cash_withdrawal_not_recognised",
    21: "change_pin",
    22: "compromised_card",
    23: "contactless_not_working",
    24: "country_support",
    25: "declined_card_payment",
    26: "declined_cash_withdrawal",
    27: "declined_transfer",
    28: "direct_debit_payment_not_recognised",
    29: "disposable_card_limits",
    30: "edit_personal_details",
    31: "exchange_charge",
    32: "exchange_rate",
    33: "exchange_via_app",
    34: "extra_charge_on_statement",
    35: "failed_transfer",
    36: "fiat_currency_support",
    37: "get_disposable_virtual_card",
    38: "get_physical_card",
    39: "getting_spare_card",
    40: "getting_virtual_card",
    41: "lost_or_stolen_card",
    42: "lost_or_stolen_phone",
    43: "order_physical_card",
    44: "passcode_forgotten",
    45: "pending_card_payment",
    46: "pending_cash_withdrawal",
    47: "pending_top_up",
    48: "pending_transfer",
    49: "pin_blocked",
    50: "receiving_money",
    51: "refund_not_showing_up",
    52: "request_refund",
    53: "reverted_card_payment",
    54: "supported_cards_and_currencies",
    55: "terminate_account",
    56: "top_up_by_bank_transfer_charge",
    57: "top_up_by_card_charge",
    58: "top_up_by_cash_or_cheque",
    59: "top_up_failed",
    60: "top_up_limits",
    61: "top_up_reverted",
    62: "topping_up_by_card",
    63: "transaction_charged_twice",
    64: "transfer_fee_charged",
    65: "transfer_into_account",
    66: "transfer_not_received_by_recipient",
    67: "transfer_timing",
    68: "unable_to_verify_identity",
    69: "verify_my_identity",
    70: "verify_source_of_funds",
    71: "verify_top_up",
    72: "virtual_card_not_working",
    73: "visa_or_mastercard",
    74: "why_verify_identity",
    75: "wrong_amount_of_cash_received",
    76: "wrong_exchange_rate_for_cash_withdrawal"
}


# ==============================
# 3. FORMAT INSTRUCTION
# ==============================
def format_example(row):
    intent_text = id2label[row["label"]]

    return f"""### Instruction:
Classify the intent of the following banking message.

### Input:
{row['text']}

### Response:
{intent_text}
"""


# ==============================
# 4. LOAD DATA
# ==============================
def load_data():
    print("Đang load dữ liệu...")

    train_df = pd.read_csv(TRAIN_FILE)

    print(f"Số lượng train: {len(train_df)}")

    train_df["text"] = train_df.apply(format_example, axis=1)

    dataset = Dataset.from_pandas(train_df)

    return dataset


# ==============================
# 5. MAIN TRAIN
# ==============================
def main():
    print("Bắt đầu training với Unsloth...")

    # Load model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Áp dụng LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
    )

    dataset = load_data()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            logging_steps=LOGGING_STEPS,
            output_dir=OUTPUT_DIR,
            save_strategy="epoch",
            report_to="none"
        ),
    )

    trainer.train()

    print("Đang lưu model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training hoàn tất!")


# ==============================
# 6. RUN
# ==============================
if __name__ == "__main__":
    main()