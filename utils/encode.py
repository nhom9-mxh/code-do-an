import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import chunks

llm = None


def init_llm(model_name: str, device: torch.device = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


@torch.inference_mode()
def forward_llm(
    llm, text: str | list[str], batch_size: int = 32, device: torch.device = "cuda"
):
    if isinstance(text, str):
        text = [text]

    model, tokenizer = llm

    output = []
    for batch in chunks(text, batch_size):
        model_inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        batch_output = model.roberta(**model_inputs).last_hidden_state[:, 0, :].cpu()
        output.append(batch_output)
    return torch.cat(output, dim=0)


def ensure_llm(fn):
    global llm
    if llm is None:
        llm = init_llm("FacebookAI/xlm-roberta-large")
    return fn


@ensure_llm
def encode_course(course: dict):
    prereq_embed = forward_llm(llm, course["prerequisites"])
    about_embed = forward_llm(llm, course["about"])

    reprentation = torch.cat([prereq_embed, about_embed], dim=-1)
    return reprentation


@ensure_llm
def encode_fields(fields: list[str]):
    if len(fields) == 0:
        return torch.empty(0, 1024, dtype=torch.float)

    return forward_llm(llm, fields)


@ensure_llm
def encode_resources(resources: list[dict]):
    if len(resources) == 0:
        return torch.empty(0, 1024, dtype=torch.float)

    titles = [resource["title"] for resource in resources]
    return forward_llm(llm, titles)


@ensure_llm
def encode_schools(schools: list[dict]):
    if len(schools) == 0:
        return torch.empty(0, 2048, dtype=torch.float)

    abouts = []
    mottos = []
    for school in schools:
        abouts.append(school["about"])
        mottos.append(school["motto"])

    about_embed = forward_llm(llm, abouts)
    motto_embed = forward_llm(llm, mottos)
    representation = torch.cat([about_embed, motto_embed], dim=-1)
    return representation


@ensure_llm
def encode_teachers(teachers: list[dict]):
    if len(teachers) == 0:
        return torch.empty(0, 2048)

    abouts = []
    job_titles = []
    for teacher in teachers:
        about = teacher["about"]
        job_title = teacher["job_title"]
        abouts.append(about)
        job_titles.append(job_title)

    abouts_embed = forward_llm(llm, abouts)
    job_titles_embed = forward_llm(llm, job_titles)
    representation = torch.cat([abouts_embed, job_titles_embed], dim=-1)
    return representation


@ensure_llm
def encode_users(users: list[dict]):
    if len(users) == 0:
        return torch.empty(0, 3, dtype=torch.float)
    genders = torch.tensor([user["gender"] for user in users])
    genders = torch.nn.functional.one_hot(genders, num_classes=3).float()
    return genders


@ensure_llm
def encode_comments(comments: list[dict]):
    if len(comments) == 0:
        return torch.empty(0, 1024, dtype=torch.float)

    texts = [comment["text"] for comment in comments]
    return forward_llm(llm, texts)


@ensure_llm
def encode_replies(replies: list[dict]):
    if len(replies) == 0:
        return torch.empty(0, 1024, dtype=torch.float)

    texts = [reply["text"] for reply in replies]
    return forward_llm(llm, texts)


@ensure_llm
def encode_exercises(exercises: list[dict]):
    if len(exercises) == 0:
        return torch.empty(0, 3072, dtype=torch.float)

    titles = []
    contents = []
    types = []
    for exercise in exercises:
        title = exercise["title"]
        content = exercise["content"]
        type = exercise["typetext"]
        titles.append(title)
        contents.append(content)
        types.append(type)

    titles_embed = forward_llm(llm, titles)
    contents_embed = forward_llm(llm, contents)
    types_embed = forward_llm(llm, types)
    representation = torch.cat([titles_embed, contents_embed, types_embed], dim=-1)
    return representation


@ensure_llm
def encode_videos(videos: list[dict]):
    if len(videos) == 0:
        return torch.empty(0, 2048, dtype=torch.float)

    names = []
    texts_embed = []
    for video in videos:
        name = video["name"]
        text_embed = forward_llm(llm, video["text"]).mean(dim=0)
        names.append(name)
        texts_embed.append(text_embed)

    names_embed = forward_llm(llm, names)
    texts_embed = torch.stack(texts_embed, dim=0)
    representation = torch.cat([names_embed, texts_embed], dim=-1)
    return representation
