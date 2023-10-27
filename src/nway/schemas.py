from argschema.schemas import DefaultSchema
from argschema import ArgSchema
from argschema.fields import InputFile, OutputDir, OutputFile
import marshmallow as mm
from marshmallow import fields
import logging
import json

logger = logging.getLogger(__name__)


class CellROISchema(DefaultSchema):
    id = fields.Int(
        description="unique ROI id. At minimum this must be unique within the "
                    "experiment container if not globally unique."
    )
    x = fields.Int(
        description="lowest x pixel coordinate of the ROI. Zero indexed"
    )
    y = fields.Int(
        description="lowest y pixel coordinate of the ROI. Zero indexed"
    )
    z = fields.Int(
        required=False,
        default=0,
        description="Unused in processing. Defaults to 0"
    )
    width = fields.Int(
        description="width of the ROI mask in pixels"
    )
    height = fields.Int(
        description="height of the ROI mask in pixels"
    )
    valid = fields.Bool(
        required=False,
        default=False,
        description="True if the ROI has no processing flags/exclusion labels "
                    "set from previous processing. Unused in this module. Any "
                    "cuts on on the set of ROIs (if desired) should be "
                    "applied while creating this input json. Defaults to "
                    "False."
    )
    mask_matrix = fields.List(
        fields.List(fields.Bool),
        description="list of lists of Booleans specifying the 2-D mask of the "
                    "individual ROI. This 2-D array plus the x, y location "
                    "specifies fully specifies the ROI mask."
    )


class OnPremExperimentSchema(DefaultSchema):
    id = fields.Int(
        description="unique experiment id. Must be at minimum unique within "
                    "the experiment container."
    )
    stimulus_name = fields.Str(
        required=False,
        allow_none=True,
        description="name of stimulus used in this experiment. Unused except "
                    "for in diagnostic plots for labeling of the plots."
    )
    ophys_average_intensity_projection_image = InputFile(
        description="path to a projection image of the ophys movie. Nominally "
                    "this is output by the suite2p_registration module of "
                    "AllenInstitute/ophys_etl_pipelines though one could """
                    "construct and use a projection of the Denoised "
                    "movie or any other kind of projection image if desired. "
                    "The image format is a one channel, unit8 PNG file."
    )
    cell_rois = fields.Nested(
        CellROISchema,
        many=True,
        description="ROIs detected in the experiment. These are produced and "
                    "formatted similarly to the "
                    "AllenInstitute/segment_postprocess module in "
                    "ophys_etl_pipelines."
    )


class ExperimentContainerSchema(DefaultSchema):
    ophys_experiments = fields.Nested(
        OnPremExperimentSchema,
        many=True,
        description="Set of ophys experiments that make up the container."
    )


class OnPremGeneratedInputSchema(DefaultSchema):
    """This structure is what comes out of the LIMS ophys_nway
    strategy
    """
    output_directory = OutputDir(
        required=False,
        description="destination for output files."
    )
    experiment_containers = fields.Nested(
        ExperimentContainerSchema,
        description="Experiment container to match ROIs across."
    )


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
                     " in pairwise matching. Will warn if Hungarian is "
                     "selected."))
    motionType = fields.Str(
        required=False,
        missing="MOTION_EUCLIDEAN",
        default="MOTION_EUCLIDEAN",
        validator=mm.validate.OneOf([
            "MOTION_TRANSLATION",
            "MOTION_EUCLIDEAN",
            "MOTION_AFFINE",
            "MOTION_HOMOGRAPHY"
        ]),
        description="What type of motion to use in cv2, ECC registration. "
    )
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
    """Class that uses argschema to take care of input arguments"""
    log_level = fields.Str(default="INFO")
    save_pairwise_results = fields.Bool(
        required=False,
        default=False,
        description="Whether to save pairwise output jsons")
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
        description="NOTE: Only for onprem use in LIMS with the legacy "
                    "segmentor. This was a temporary hack. If set to true, "
                    "module will attempt to substitute 'maxInt_a13a.png' for "
                    "'avgInt_a1X.png' in the "
                    "'ophys_average_intensity_projection_image' fields of "
                    "the input experiments. The registration is attempted "
                    "on max rather than avg projections. "
    )


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
    scale = fields.Tuple(
        (fields.Float(), fields.Float()),
        description="tuple of floats specifying the scale in x and y"
    )
    shear = fields.Float(
        description="amount of shear in the affine transformation matrix"
    )
    rotation = fields.Float(
        description="rotation angle in radians."
    )
    translation = fields.Tuple(
        (fields.Float(), fields.Float()),
        description="tuple of floats specifying the translation in x and y in "
                    "pixels"
    )


class TransformSchema(DefaultSchema):
    best_registration = fields.List(
        fields.Str,
        description="list of strings specifying the transformations used for "
                    "the best registration."
    )
    properties = fields.Nested(
        TransformPropertySchema,
        description="decomposed properties for the approximate "
                    "affine transformation matrix."
    )
    matrix = fields.List(
        fields.List(fields.Float()),
        description="Full, approximate affine transformation matrix."
    )
    transform_type = fields.Str(
        description="type of transformation used for ECC registration. Will "
                    "be one of MOTION_TRANSLATION, MOTION_EUCLIDEAN, "
                    "MOTION_AFFINE, MOTION_HOMOGRAPHY as specified in the "
                    "input schema parameter `motionType`."
    )


class MatchSchema(DefaultSchema):
    fixed = fields.Int(
        description="id of ROI from fixed experiment"
    )
    moving = fields.Int(
        description="id of ROI from moving experiment"
    )
    distance = fields.Float(
        description="distance between ROIs normalized by the value of "
                    "`maximum_distance`."
    )
    cost = fields.Float(
        description="cost of the match as calculated in the method "
                    "calculate_cost_matrix in pairwarise_matching.py"
    )
    iou = fields.Float(
        description="intersection over union of the matched ROI masks"
    )


class UnmatchedSchema(DefaultSchema):
    fixed = fields.List(
        fields.Int(),
        required=False,
        description="list of unmatched ROI ids from the fixed_experiment"
    )
    moving = fields.List(
        fields.Int(),
        required=False,
        description="list of unmatched ROI ids from the moving_experiment"
    )


class PairwiseOutputSchema(DefaultSchema):
    fixed_experiment = fields.Int(
        required=True,
        description="fixed experiment id in pair")
    moving_experiment = fields.Int(
        required=True,
        description="moving experiment id in pair")
    transform = fields.Nested(
        TransformSchema,
        description="approximate affine transformation from moving to fixed "
                    "experiment."
    )
    matches = fields.Nested(
        MatchSchema,
        many=True,
        description="Full summary of each ROI match."
    )
    rejected = fields.Nested(
        MatchSchema,
        many=True,
        description="Full summary of each rejected ROI match (e.g. values "
                    "calculated but distance is outside of )."
    )
    unmatched = fields.Nested(
        UnmatchedSchema,
        description="Full summary of unmatched ROIs."
    )


class NwayMatchingOutputNoPlotsSchema(DefaultSchema):
    nway_matches = fields.List(
        fields.List(fields.Int),
        required=True,
        description="list of lists of matching IDs")
    pairwise_results = fields.Nested(
        PairwiseOutputSchema,
        many=True,
        description="Full summary of each pairwise match result."
    )


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
