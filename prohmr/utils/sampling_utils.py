import numpy as np

from prohmr.utils.pose_utils import scale_and_translation_transform_batch


def compute_vertex_uncertainties_from_samples(vertices_samples,
                                              target_vertices=None,
                                              return_separate_reposed_dims=False):
    num_samples = vertices_samples.shape[0]

    mean_vertices = vertices_samples.mean(axis=0)
    avg_vertices_distance_from_mean = np.linalg.norm(vertices_samples - mean_vertices, axis=-1).mean(axis=0)  # (6890,)
    if return_separate_reposed_dims:
        avg_vertices_distance_from_mean_xyz = np.mean(np.sqrt(np.square(vertices_samples - mean_vertices)), axis=0)  # (6890, 3)
    if target_vertices is not None:
        target_vertices = np.concatenate([target_vertices] * num_samples, axis=0)  # (num_samples, 6890, 3)
        vertices_sc_samples = scale_and_translation_transform_batch(vertices_samples, target_vertices)
        mean_vertices_sc = vertices_sc_samples.mean(axis=0)
        avg_vertices_sc_distance_from_mean = np.linalg.norm(vertices_sc_samples - mean_vertices_sc, axis=-1).mean(axis=0)  # (6890,)
    else:
        avg_vertices_sc_distance_from_mean = None
    if return_separate_reposed_dims:
        return avg_vertices_distance_from_mean, avg_vertices_sc_distance_from_mean, avg_vertices_distance_from_mean_xyz
    else:
        return avg_vertices_distance_from_mean, avg_vertices_sc_distance_from_mean