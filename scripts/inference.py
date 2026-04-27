# ==============================
# inference.py
# ==============================

import yaml
import torch
from unsloth import FastLanguageModel

# ==============================
# LABEL MAP
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

class IntentClassification:

    def __init__(self, model_path):
        """
        model_path: đường dẫn tới file config YAML
        """

        print("Đang load config...")

        with open(model_path, "r") as f:
            config = yaml.safe_load(f)

        self.base_model = config["base_model"]
        self.adapter_model = config["adapter_model"]
        self.max_seq_length = config["max_seq_length"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Sử dụng device: {self.device}")

        print("Đang load base model...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=(self.device == "cuda"),
        )

        print("Đang load adapter...")

        self.model.load_adapter(self.adapter_model)
        self.model.eval()

        print("Load model hoàn tất!")


    def __call__(self, message):
        """
        message: input text
        return: predicted label (string)
        """

        prompt = f"""### Instruction:
Classify the intent of the following banking message.
Only return the intent name.

### Input:
{message}

### Response:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            max_length=None,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Response:" in output_text:
            output_text = output_text.split("### Response:")[-1]

        output_text = output_text.strip().split("\n")[0]

        return output_text.strip()


# ==============================
# USAGE EXAMPLE 
# ==============================
if __name__ == "__main__":

    model = IntentClassification("configs/inference.yaml")

    texts = [
        "I lost my card",
        "My payment didn't go through",
        "How to activate my card?",
        "I want to transfer money",
        "Why is my balance not updated?",
        "Can I use Apple Pay?",
        "My card was stolen",
        "I can't withdraw cash",
        "How to check my account balance?",
        "Cancel my transfer please"
    ]

    for text in texts:
        result = model(text)

        print("\nInput :", text)
        print("Output:", result)
        print("-" * 40)