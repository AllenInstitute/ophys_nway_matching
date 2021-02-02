import argschema
from nway.assembly_utils import query_utils


class InputAssemblyInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(default="INFO")
    experiment_ids = argschema.fields.List(
        argschema.fields.Int,
        required=True,
        cli_as_single_argument=True,
        description=("list of experiment IDs. These will be queried to "
                     "assemble an input_json"))


class InputAssembly(argschema.ArgSchemaParser):
    default_schema = InputAssemblyInputSchema

    def run(self, dbconn):
        qstring = "SELECT id FROM ophys_experiments WHERE id IN ({})"
        str_ids = ",".join([str(i) for i in self.args['experiment_ids']])
        results = dbconn.query(qstring.format(str_ids))
        self.logger.info(results)


if __name__ == "__main__":
    lims_credentials = query_utils.get_db_credentials()
    lims_connection = query_utils.DbConnection(**lims_credentials)
    ia = InputAssembly()
    ia.run(lims_connection)
