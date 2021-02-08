from argschema.schemas import DefaultSchema
from argschema import ArgSchema
from argschema.fields import InputFile, OutputDir, OutputFile
import marshmallow as mm
from marshmallow import fields
import logging
import json

logger = logging.getLogger(__name__)


class CellROISchema(DefaultSchema):
    id = fields.Int()
    x = fields.Int()
    y = fields.Int()
    z = fields.Int()
    width = fields.Int()
    height = fields.Int()
    valid = fields.Bool()
    mask_matrix = fields.List(fields.List(fields.Bool))


class OnPremExperimentSchema(DefaultSchema):
    id = fields.Int()
    stimulus_name = fields.Str(required=False, allow_none=True)
    ophys_average_intensity_projection_image = InputFile()
    cell_rois = fields.Nested(CellROISchema, many=True)


class ExperimentContainerSchema(DefaultSchema):
    ophys_experiments = fields.Nested(OnPremExperimentSchema, many=True)


class OnPremGeneratedInputSchema(DefaultSchema):
    """This structure is what comes out of the LIMS ophys_nway
    strategy
    """
    output_directory = OutputDir(required=False)
    experiment_containers = fields.Nested(ExperimentContainerSchema)


class PairwiseExperimentSchema(OnPremExperimentSchema):
    nice_mask_path = InputFile(
        required=True,
        description="path to mask tiff with unique labels")
    nice_dict_path = InputFile(
        required=True,
        description="path to dict for mask labels to LIMS ids")


class CommonMatchingParameters(DefaultSchema):
    maximum_distance = fields.Int(
        required=False,
        default=10,
        description=("Maximum distance (in pixels) between two cells,"
                     " above which a match is always rejected."))
    registration_iterations = fields.Int(
        required=False,
        default=1000,
        description=("Number of iterations for intensity-"
                     "based registration"))
    registration_precision = fields.Float(
        required=False,
        default=1.5e-7,
        description=("Threshold of squared error, below which "
                     "registration is terminated"))
    assignment_solver = fields.Str(
        required=False,
        default="Blossom",
        missing="Blossom",
        validator=mm.validate.OneOf([
            "Blossom",
            "Hungarian"]),
        description=("What method to use for solving the assignment problem"
                     " in pairwise matching"))
    motionType = fields.Str(
        required=False,
        missing="MOTION_EUCLIDEAN",
        default="MOTION_EUCLIDEAN",
        validator=mm.validate.OneOf([
            "MOTION_TRANSLATION",
            "MOTION_EUCLIDEAN",
            "MOTION_AFFINE",
            "MOTION_HOMOGRAPHY"
            ]))
    gaussFiltSize = fields.Int(
        required=False,
        missing=5,
        default=5,
        description="passed to opencv findTransformECC")
    CLAHE_grid = fields.Int(
        required=False,
        default=24,
        missing=24,
        description="tileGridSize for cv2 CLAHE, set to -1 to disable CLAHE")
    CLAHE_clip = fields.Float(
        required=False,
        default=2.5,
        missing=2.5,
        description="clipLimit for cv2 CLAHE")
    edge_buffer = fields.Int(
        required=False,
        default=40,
        missing=40,
        description=("in the Crop transform for meta-registration, this "
                     "many pixels will be cropped from the top, bottom, "
                     "left, and right of both images."))
    include_original = fields.Bool(
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


class NwayMatchingSchema(ArgSchema,
                         CommonMatchingParameters,
                         OnPremGeneratedInputSchema):
    ''' Class that uses argschema to take care of input arguments '''
    log_level = fields.Str(default="INFO")
    save_pairwise_results = fields.Bool(
        required=False,
        default=False,
        description=("Whether to save pairwise output jsons"))
    pruning_method = fields.Str(
        required=False,
        missing="keepmin",
        default="keepmin",
        description=("method for reducing pruning graph: "
                     "'keepmin': similar to original, delete from "
                     "    graph neighbors of lowest subgraph score."
                     "'popmax': delete highest scores of subgraphs "
                     "    recursively"))
    parallel_workers = fields.Int(
        required=False,
        missing=3,
        default=3,
        description=("number of parallel workers for multiprocessing pool "
                     "for executing pairwise matching. If 1, runs serially."))
    substitute_max_for_avg = fields.Bool(
        required=False,
        missing=False,
        default=False,
        description=("if set to true, module will attempt to substitute "
                     "'maxInt_a13a.png' for 'avgInt_a1X.png' in the "
                     "'ophys_average_intensity_projection_image' fields of "
                     "the input experiments. The registration is attempted "
                     "on max rather than avg projections. "
                     "NOTE: there is a better way to "
                     "accomplish this. This is a temporary hack."))


class PairwiseMatchingSchema(ArgSchema, CommonMatchingParameters):
    output_directory = OutputDir(
        required=True,
        description="destination for output files.")
    save_registered_image = fields.Bool(
        required=False,
        default=False,
        description='Whether to save registered image.')
    fixed = fields.Nested(PairwiseExperimentSchema)
    moving = fields.Nested(PairwiseExperimentSchema)


class TransformPropertySchema(DefaultSchema):
    scale = fields.Tuple((fields.Float(), fields.Float()))
    shear = fields.Float()
    rotation = fields.Float()
    translation = fields.Tuple((fields.Float(), fields.Float()))


class TransformSchema(DefaultSchema):
    best_registration = fields.List(fields.Str)
    properties = fields.Nested(TransformPropertySchema)
    matrix = fields.List(fields.List(fields.Float()))
    transform_type = fields.Str()


class MatchSchema(DefaultSchema):
    fixed = fields.Int()
    moving = fields.Int()
    distance = fields.Float()
    cost = fields.Float()
    iou = fields.Float()


class UnmatchedSchema(DefaultSchema):
    fixed = fields.List(fields.Int(), required=False)
    moving = fields.List(fields.Int(), required=False)


class PairwiseOutputSchema(DefaultSchema):
    fixed_experiment = fields.Int(
        required=True,
        description="fixed experiment id of pair")
    moving_experiment = fields.Int(
        required=True,
        description="moving experiment id of pair")
    transform = fields.Nested(TransformSchema)
    matches = fields.Nested(MatchSchema, many=True)
    rejected = fields.Nested(MatchSchema, many=True)
    unmatched = fields.Nested(UnmatchedSchema)


class NwayMatchingOutputNoPlotsSchema(DefaultSchema):
    nway_matches = fields.List(
        fields.List(fields.Int),
        required=True,
        description="list of lists of matching IDs")
    pairwise_results = fields.Nested(PairwiseOutputSchema, many=True)


class NwayMatchingOutputSchema(NwayMatchingOutputNoPlotsSchema):
    nway_match_fraction_plot = OutputFile(
        required=True,
        description="Path of match fraction plot *.png")
    nway_warp_overlay_plot = OutputFile(
        required=True,
        description="Path of warp overlay plot *.png")
    nway_warp_summary_plot = OutputFile(
        required=True,
        description="Path of warp summary plot *.png")


class NwayDiagnosticSchema(ArgSchema):
    output_pdf = OutputFile(
        required=True,
        description="path to output pdf")
    use_input_dir = fields.Bool(
        required=False,
        missing=False,
        default=False,
        descriptip="output to same directory as input")


class NwayMatchSummarySchema(ArgSchema):
    input_file = InputFile(
        required=False,
        desc="Input *.json file path to nway matching."
    )
    output_file = InputFile(
        required=False,
        desc="Output *.json file from nway matching."
    )
    # TODO: eliminate this non-specific Dict
    nway_input = fields.Dict(
        required=True,
        desc="Input to nway matching in Python dictionary form."
    )
    nway_output = fields.Nested(NwayMatchingOutputNoPlotsSchema)
    output_directory = OutputDir(
        required=True,
        description="Destination for summary plot output file(s).")

    @mm.pre_load
    def fill_dict_inputs(self, data: dict, **kwargs) -> dict:
        if not data['nway_input']:
            with open(data['input_file'], 'r') as f:
                input_dict = json.load(f)
            data['nway_input'] = input_dict
        elif data.get('input_file'):
            logger.warning("Both --nway_input and --input_file were provided "
                           "so --input_file will be ignored.")

        if not data['nway_output']:
            with open(data['output_file'], 'r') as f:
                output_dict = json.load(f)
            data['nway_output'] = output_dict
        elif data.get('output_file'):
            logger.warning("Both --nway_output and --output_file were "
                           "provided so --output_file will be ignored.")

        return data
