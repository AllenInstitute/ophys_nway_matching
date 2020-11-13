from argschema.schemas import DefaultSchema
from argschema import ArgSchema
from argschema.fields import (
        Boolean, Int, Str, Float, Dict,
        List, InputFile, OutputDir, Nested,
        OutputFile, Bool)
import marshmallow as mm
import logging

logger = logging.getLogger(__name__)


class ExperimentSchema(DefaultSchema):
    id = Int(
        required=True,
        description="experiment id")
    ophys_average_intensity_projection_image = InputFile(
        required=True,
        description="max projection intensity image")
    cell_rois = List(
        Dict,
        required=True,
        cli_as_single_argument=True,
        description='dict mapping of ids, labels, zs,')
    nice_mask_path = InputFile(
        required=True,
        description="path to mask tiff with unique labels")
    nice_dict_path = InputFile(
        required=True,
        description="path to dict for mask labels to LIMS ids")


class CommonMatchingSchema(ArgSchema):
    log_level = Str(
        default="INFO",
        description="override argschema default")
    maximum_distance = Int(
        required=False,
        default=10,
        description=("Maximum distance (in pixels) between two cells,"
                     " above which a match is always rejected."))
    registration_iterations = Int(
        required=False,
        default=1000,
        description=("Number of iterations for intensity-"
                     "based registration"))
    registration_precision = Float(
        required=False,
        default=1.5e-7,
        description=("Threshold of squared error, below which "
                     "registration is terminated"))
    assignment_solver = Str(
        required=False,
        default="Blossom",
        missing="Blossom",
        validator=mm.validate.OneOf([
            "Blossom",
            "Hungarian"]),
        description=("What method to use for solving the assignment problem"
                     " in pairwise matching"))
    motionType = Str(
        required=False,
        missing="MOTION_EUCLIDEAN",
        default="MOTION_EUCLIDEAN",
        validator=mm.validate.OneOf([
            "MOTION_TRANSLATION",
            "MOTION_EUCLIDEAN",
            "MOTION_AFFINE",
            "MOTION_HOMOGRAPHY"
            ]))
    gaussFiltSize = Int(
        required=False,
        missing=5,
        default=5,
        description="passed to opencv findTransformECC")
    CLAHE_grid = Int(
        required=False,
        default=24,
        missing=24,
        description="tileGridSize for cv2 CLAHE, set to -1 to disable CLAHE")
    CLAHE_clip = Float(
        required=False,
        default=2.5,
        missing=2.5,
        description="clipLimit for cv2 CLAHE")
    edge_buffer = Int(
        required=False,
        default=40,
        missing=40,
        description=("in the Crop transform for meta-registration, this "
                     "many pixels will be cropped from the top, bottom, "
                     "left, and right of both images."))
    include_original = Bool(
        required=False,
        default=False,
        missing=False,
        description=("whether to include the original registration strategy "
                     "which occasionally succeeds with a not great result."))

    @mm.post_load
    def hungarian_warn(self, data, **kwargs):
        if "Hungarian" in data['assignment_solver']:
            logger.warning("Hungarian method not recommended. It is not "
                           "stable under permutations for typical cell "
                           "matching. Use Blossom.")
        return data


class NwayMatchingSchema(CommonMatchingSchema):
    ''' Class that uses argschema to take care of input arguments '''
    output_directory = OutputDir(
        required=True,
        description=("destination for output files. If None, will be set from "
                     "output_directory field in input file"))
    experiment_containers = Dict(
        required=True,
        description="contains data for matching")
    save_pairwise_results = Boolean(
        required=False,
        default=False,
        description=("Whether to save pairwise output jsons"))
    pruning_method = Str(
        required=False,
        missing="keepmin",
        default="keepmin",
        description=("method for reducing pruning graph: "
                     "'keepmin': similar to original, delete from "
                     "    graph neighbors of lowest subgraph score."
                     "'popmax': delete highest scores of subgraphs "
                     "    recursively"))
    parallel_workers = Int(
        required=False,
        missing=3,
        default=3,
        description=("number of parallel workers for multiprocessing pool "
                     "for executing pairwise matching. If 1, runs serially."))


class PairwiseMatchingSchema(CommonMatchingSchema):
    output_directory = OutputDir(
        required=True,
        description="destination for output files.")
    save_registered_image = Boolean(
        required=False,
        default=True,
        description='Whether to save registered image.')
    fixed = Nested(ExperimentSchema)
    moving = Nested(ExperimentSchema)


class NwayMatchingOutputSchema(DefaultSchema):
    nway_matches = List(
        List(Int),
        required=True,
        description="list of lists of matching IDs")
    pairwise_results = List(
        Dict,
        required=True,
        description="list of pairwise result dicts")


class PairwiseOutputSchema(DefaultSchema):
    fixed_experiment = Int(
        required=True,
        description="fixed experiment id of pair")
    moving_experiment = Int(
        required=True,
        description="moving experiment id of pair")
    transform = Dict(
        required=True,
        description="transform applied to moving")
    matches = List(
        Dict,
        required=True,
        description="matches made from pairwise matching")
    rejected = List(
        Dict,
        required=True,
        description="pairs within max_distance, but not matched")
    unmatched = Dict(
        required=True,
        description="list of cell IDs that were never in a considered pair")


class NwayDiagnosticSchema(ArgSchema):
    output_pdf = OutputFile(
        required=True,
        description="path to output pdf")
    use_input_dir = Bool(
        required=False,
        missing=False,
        default=False,
        descriptip="output to same directory as input")
