import torch


def course_field_edges(fields: list[str]):
    n_fields = len(fields)
    return torch.stack([torch.zeros(n_fields), torch.arange(n_fields)]).long()


def course_resource_edges(resources: list[dict]):
    n_resources = len(resources)
    return torch.stack([torch.zeros(n_resources), torch.arange(n_resources)]).long()


def course_teacher_edges(teachers: list[dict]):
    n_teachers = len(teachers)
    return torch.stack([torch.zeros(n_teachers), torch.arange(n_teachers)]).long()


def course_school_edges(course: dict, schools: list[dict]):
    school_id = course["school_id"]
    school_idx = [school["_id"] for school in schools].index(school_id)
    return torch.tensor([[0], [school_idx]]).long()


def course_comment_edges(comments: list[dict]):
    n_comments = len(comments)
    return torch.stack([torch.zeros(n_comments), torch.arange(n_comments)]).long()


def course_user_edges(users: list[dict]):
    n_users = len(users)
    return torch.stack([torch.zeros(n_users), torch.arange(n_users)]).long()


def comment_reply_edges(comments: list[dict], replies: list[dict]):
    cmt2idx = {comment["_id"]: i for i, comment in enumerate(comments)}
    reply2idx = {reply["_id"]: i for i, reply in enumerate(replies)}
    edges = []
    for reply in replies:
        comment_id = reply["comment_id"]
        if comment_id in cmt2idx:
            edges.append([cmt2idx[comment_id], reply2idx[reply["_id"]]])

    return torch.tensor(edges).T.long()


def user_comment_edges(users: list[dict], comments: list[dict]):
    cmt2idx = {comment["_id"]: i for i, comment in enumerate(comments)}
    user2idx = {user["_id"]: i for i, user in enumerate(users)}

    edges = []
    for comment in comments:
        user_id = comment["user_id"]
        if user_id in user2idx:
            edges.append([user2idx[user_id], cmt2idx[comment["_id"]]])

    return torch.tensor(edges).T.long()


def user_reply_edges(users: list[dict], replies: list[dict]):
    reply2idx = {reply["_id"]: i for i, reply in enumerate(replies)}
    user2idx = {user["_id"]: i for i, user in enumerate(users)}

    edges = []
    for reply in replies:
        user_id = reply["user_id"]
        if user_id in user2idx:
            edges.append([user2idx[user_id], reply2idx[reply["_id"]]])

    return torch.tensor(edges).T.long()


def school_user_edges(schools: list[dict], users: list[dict]):
    school2idx = {school["name"]: i for i, school in enumerate(schools)}
    user2idx = {user["school"]: i for i, user in enumerate(users)}

    edges = []
    for user in users:
        school_name = user["school"]
        if school_name in school2idx:
            edges.append([school2idx[school_name], user2idx[user["school"]]])

    return torch.tensor(edges).T.long()


def school_teacher_edges(schools: list[dict], teachers: list[dict]):
    school2idx = {school["name"]: i for i, school in enumerate(schools)}
    teacher2idx = {teacher["org_name"]: i for i, teacher in enumerate(teachers)}

    edges = []
    for teacher in teachers:
        school_name = teacher["org_name"]
        if school_name in school2idx:
            edges.append([school2idx[school_name], teacher2idx[teacher["org_name"]]])

    return torch.tensor(edges).T.long()


def resource_exercise_edges(resources: list[dict], exercises: list[dict]):
    resource2idx = {resource["_id"]: i for i, resource in enumerate(resources)}
    problem2idx = {problem["exercise_id"]: i for i, problem in enumerate(exercises)}

    edges = []
    for exercise in exercises:
        resource_id = exercise["exercise_id"]
        if resource_id in resource2idx:
            edges.append(
                [resource2idx[resource_id], problem2idx[exercise["exercise_id"]]]
            )

    return torch.tensor(edges).T.long()


def resource_video_edges(resources: list[dict], videos: list[dict]):
    resource2idx = {
        resource["ccid"]: i
        for i, resource in enumerate(resources)
        if "ccid" in resource
    }
    video2idx = {video["_id"]: i for i, video in enumerate(videos)}

    edges = []
    for resource in resources:
        if "ccid" not in resource:
            continue

        re_ccid = resource["ccid"]
        if re_ccid in video2idx:
            edges.append([resource2idx[re_ccid], video2idx[resource["ccid"]]])

    return torch.tensor(edges).T.long()
