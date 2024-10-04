import os

import numpy as np
import torch
from tqdm import tqdm


def read_hidden_states_from_path(
    path,
    index_img=None,
    index_txt=None,
    hidden_states_key=["hidden_states"],
    prompt_len=10,
    nb_elements=200,
    skip_last_txt_tokens=0,
):
    data = torch.load(path)
    feats = data["all_hidden_states"][:nb_elements]
    all_image_embeds_before = []
    all_image_embeds = []
    all_text_embeds = []
    all_image_embeds_connector = []

    print(f"reading {path}, prompt_len: {prompt_len}")

    for hskey in hidden_states_key:

        image_embeds_before = []
        image_embeds = []
        text_embeds = []
        image_embeds_connector = []
        for d in tqdm(feats):
            hidden_states = d[hskey]  # 32, l, d
            att = d["attention_masks"].numpy()

            if "vision_states_before" in d:
                im_before = d["vision_states_before"]
                im_before = im_before.numpy()

                if index_img is not None:
                    im_before = im_before[index_img]
                else:
                    im_before = im_before.mean(0)
            else:
                im_before = None

            image_embeds_before.append(im_before)

            if "vision_states_connector" in d:
                im_before = d["vision_states_connector"]

                im_befores = []
                for hs in im_before:
                    hs = hs.numpy()
                    if index_img is not None:
                        hs = hs[index_img]
                    else:
                        hs = hs.mean(0)
                    im_befores.append(hs)
            else:
                im_befores = None

            image_embeds_connector.append(im_befores)

            ts = []
            ims = []

            for i, hs in enumerate(hidden_states):
                if hs.ndim < 2:
                    continue
                hs = hs.numpy()

                t = hs[prompt_len:, :]
                im = hs[:prompt_len, :]
                if att.shape[0] == t.shape[0]:
                    t = t[att.astype(bool)]

                if skip_last_txt_tokens > 0:
                    t = t[:-skip_last_txt_tokens, :]

                if index_txt is not None:
                    if index_txt >= 0:
                        t = t[index_txt]
                else:
                    t = t.mean(0)

                if index_img is not None:
                    if index_img >= 0:
                        im = im[index_img]
                else:
                    im = im.mean(0)

                ts.append(t)
                ims.append(im)

            text_embeds.append(ts)

            image_embeds.append(ims)

        all_image_embeds.append(image_embeds)
        all_text_embeds.append(text_embeds)
        all_image_embeds_before.append(image_embeds_before)
        all_image_embeds_connector.append(image_embeds_connector)

    if len(hidden_states_key) == 1:
        (
            all_image_embeds,
            all_text_embeds,
            all_image_embeds_before,
            all_image_embeds_connector,
        ) = (
            all_image_embeds[0],
            all_text_embeds[0],
            all_image_embeds_before[0],
            all_image_embeds_connector[0],
        )

    return (
        all_image_embeds,
        all_text_embeds,
        all_image_embeds_before,
        all_image_embeds_connector,
    )


def read_hidden_states_from_folders(
    results_dir,
    folders,
    file_name,
    labels,
    index_img,
    index_txt,
    hidden_states_key=["hidden_states"],
    prompt_len=10,
    skip_last_txt_tokens=0,
):

    new_labels = []
    all_embed_img = []
    all_embed_txt = []
    all_embed_img_before = []
    all_embed_img_connector = []

    for i, folder in enumerate(folders):
        path = os.path.join(results_dir, folder, file_name)
        if not isinstance(index_img, list):
            index_img = [index_img]

        imgs = []
        for index_img_ in index_img:
            (
                image_embeds,
                text_embeds,
                image_embeds_before,
                image_embeds_connector,
            ) = read_hidden_states_from_path(
                path,
                index_img=index_img_,
                index_txt=index_txt,
                hidden_states_key=hidden_states_key,
                prompt_len=prompt_len,
                skip_last_txt_tokens=skip_last_txt_tokens,
            )  # n, 32, d
            new_labels.append(labels[i])
            imgs += image_embeds

        all_embed_txt.append(text_embeds)
        all_embed_img.append(imgs)
        all_embed_img_before.append(image_embeds_before)
        all_embed_img_connector.append(image_embeds_connector)

    print("finish reading", len(image_embeds))

    return (
        all_embed_txt,
        all_embed_img,
        new_labels,
        all_embed_img_before,
        all_embed_img_connector,
    )


IMAGE_TOKEN_INDEX = -200


def read_hidden_states_from_path_llava(
    path,
    index_img=None,
    index_txt=None,
    hidden_states_key=["hidden_states"],
    prompt_len=576,
    nb_elements=200,
    discard_first_text=False,
):
    data = torch.load(path)
    feats = data["all_hidden_states"][:nb_elements]
    all_image_embeds = []
    all_text_embeds = []

    print(f"reading {path}, prompt_len: {prompt_len}")

    for hskey in hidden_states_key:

        image_embeds = []
        text_embeds = []
        for d in tqdm(feats):

            hidden_states = d[hskey]  # 32, l, d
            att = d["attention_masks"].numpy()
            input_ids = d["input_ids"]

            ts = []
            ims = []

            if IMAGE_TOKEN_INDEX not in input_ids:
                continue

            img_token_id = np.where(input_ids == IMAGE_TOKEN_INDEX)[0][0]
            start_img_id = img_token_id
            end_img_id = img_token_id + prompt_len

            for i, hs in enumerate(hidden_states):
                if hs.ndim < 2:
                    continue
                hs = hs.numpy()

                if prompt_len > 1:
                    if discard_first_text:
                        t = hs[end_img_id:, :]
                        att = att[start_img_id + 1 :]
                    else:
                        t = np.concatenate((hs[0:start_img_id, :], hs[end_img_id:, :]))
                        att = np.concatenate(
                            (att[0:start_img_id], att[start_img_id + 1 :])
                        )
                    im = hs[start_img_id:end_img_id, :]
                else:
                    im, t = hs[:1], hs[1:]

                if att.shape[0] == t.shape[0]:
                    t = t[att.astype(bool)]

                if index_txt is not None:
                    if index_txt > 0:
                        t = t[index_txt]
                else:
                    t = t.mean(0)

                if index_img is not None:
                    if index_img > 0:
                        im = im[index_img]
                else:
                    im = im.mean(0)

                ts.append(t)
                ims.append(im)

            text_embeds.append(ts)

            image_embeds.append(ims)

        all_image_embeds.append(image_embeds)
        all_text_embeds.append(text_embeds)

    print(im.shape, t.shape, hs[start_img_id:end_img_id, :].shape)
    if len(hidden_states_key) == 1:
        all_image_embeds, all_text_embeds = all_image_embeds[0], all_text_embeds[0]

    del data
    del feats
    return all_image_embeds, all_text_embeds


def read_hidden_states_from_folders_llava(
    results_dir,
    folders,
    file_name,
    labels,
    index_img,
    index_txt,
    hidden_states_key=["hidden_states"],
    prompt_len=576,
    discard_first_text=False,
):

    new_labels = []
    all_embed_img = []
    all_embed_txt = []

    print(labels, folders)
    for i, folder in enumerate(folders):
        path = os.path.join(results_dir, folder, file_name)
        if not isinstance(index_img, list):
            index_img = [index_img]

        imgs = []
        for index_img_ in index_img:
            image_embeds, text_embeds = read_hidden_states_from_path_llava(
                path,
                index_img=index_img_,
                index_txt=index_txt,
                hidden_states_key=hidden_states_key,
                prompt_len=prompt_len,
                discard_first_text=discard_first_text,
            )  # n, 32, d
            new_labels.append(labels[i])

            imgs += image_embeds

        all_embed_txt.append(text_embeds)
        all_embed_img.append(imgs)

    return all_embed_txt, all_embed_img, new_labels
