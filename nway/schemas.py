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
    max_int_mask_image = InputFile(
        required=True,
        description="mask image")
    cell_rois = List(
        Dict,
        required=True,
        description='dict mapping of ids, labels, zs,')
    nice_mask_path = InputFile(
        required=True,
        description="path to mask tiff with unique labels")
    nice_dict_path = InputFile(
        required=True,
        description="path to dict for mask labels to LIMS ids")


class CommonMatchingSchema(ArgSchema):
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
    hungarian_executable = InputFile(
        required=False,
        missing=None,
        default=None,
        description=("Executable of Hungarian algorithm for bipartite"
                     "graph matching."))
    assignment_solver = Str(
        required=False,
        default="Blossom",
        missing="Blossom",
        validator=mm.validate.OneOf([
            "Blossom",
            "Hungarian",
            "Hungarian-cpp"]),
        description=("What method to use for solving the assignment problem"
                     " in pairwise matching"))
    motionType = Str(
        required=False,
        missing="MOTION_AFFINE",
        default="MOTION_AFFINE",
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
    integer_centroids = Boolean(
        required=False,
        default=False,
        missing=False,
        description="round ROI centroids to integer values")
    iou_flooring = Boolean(
        required=False,
        default=False,
        missing=False,
        description="use legacy '//' that sets almost all IOU's to zero")
    legacy = Boolean(
        required=False,
        default=False,
        missing=False,
        description=("Establishes 6 settings to reproduce legacy results. "
                     "3 settings are in pairwise and 3 settings are in nway. "
                     "NOTE: some of these legacy settings are mistakes. "))
    CLAHE_grid = Int(
        required=False,
        default=8,
        missing=8,
        description="tileGridSize for cv2 CLAHE, set to -1 to disable CLAHE")
    CLAHE_clip = Float(
        required=False,
        default=2.5,
        missing=2.5,
        description="clipLimit for cv2 CLAHE")

    @mm.pre_load
    def set_common_legacy(self, data):
        if data['legacy']:
            data['integer_centroids'] = True
            data['iou_flooring'] = True
            data['assignment_solver'] = 'Hungarian-cpp'
            data['CLAHE_grid'] = -1

    @mm.post_load
    def hungarian_warn(self, data):
        if "Hungarian" in data['assignment_solver']:
            logger.warning("Hungarian methid not recommended. It is not "
                           "stable under permutations for typical cell "
                           "matching. Use Blossom.")

    @mm.post_load
    def check_exe(self, data):
        if data['assignment_solver'] == 'Hungarian-cpp':
            if data['hungarian_executable'] is None:
                raise mm.ValidationError(
                    "one must supply an executable path to "
                    "--hungarian_executable when specifying "
                    " --assignment_solver Hungarian-cpp")


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
    legacy_label_order = Boolean(
        required=False,
        default=False,
        missing=False,
        description=("insist on per-layer mask label ordering created by "
                     "skimage.measure.label. Does not matter, except for "
                     "legacy Hungarian assignment."))
    legacy_pruning_order_dependence = Boolean(
        required=False,
        default=False,
        missing=False,
        description=("use original nway-pruning logic which is "
                     "order-dependent"))
    legacy_pruning_index_error = Boolean(
        required=False,
        default=False,
        missing=False,
        description="preserve index error in nway-pruning")
    pruning_method = Str(
        required=False,
        missing="keepmin",
        default="keepmin",
        description=("method for reducing pruning graph: "
                     "'keepmin': similar to original, delete from "
                     "    graph neighbors of lowest subgraph score."
                     "'popmax': delete highest scores of subgraphs "
                     "    recursively"))

    @mm.pre_load
    def set_nway_legacy(self, data):
        if data['legacy']:
            data['legacy_label_order'] = True
            data['legacy_pruning_index_error'] = True
            data['legacy_pruning_order_dependence'] = True


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


class NwayDiagnosticSchema(ArgSchema):
    output_pdf = OutputFile(
        required=True,
        description="path to output pdf")
    use_input_dir = Bool(
        required=False,
        missing=False,
        default=False,
        descriptip="output to same directory as input")
