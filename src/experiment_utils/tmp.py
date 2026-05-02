from datasets import load_dataset
from PIL import Image
# For OK-VQA
ds = load_dataset("lmms-lab/OK-VQA")

ds = ds["val2014"]

print(len(ds))


example = ds[10]


print("Question ID:", example["question_id"])
print("Question:", example["question"])
print("Answers:", example["answers"])
print("Image path:", example["image"])


# Show the image
example["image"].show()
