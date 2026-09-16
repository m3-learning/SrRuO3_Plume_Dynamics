from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataerai_notebook import DisplayOutput, NotebookProvenance, SCHEMA


USER_ID = "11111111-1111-4111-8111-111111111111"
PROJECT_ID = "22222222-2222-4222-8222-222222222222"
COLLECTION_ID = "33333333-3333-4333-8333-333333333333"


class FakeShell:
    def __init__(self) -> None:
        self.user_ns = {}
        self.registered = None
        self.kernel = None

    def register_magic_function(self, function, *, magic_kind, magic_name):
        self.registered = (function, magic_kind, magic_name)

    @staticmethod
    def transform_cell(source: str) -> str:
        return source


class FakeClient:
    def __init__(self, *, user_id: str | None = USER_ID) -> None:
        self.uploads = []
        self.relationships = []
        self.metadata_updates = []
        self.record_types = {}
        self.projects = []
        self.closed = False
        self.user_id = user_id

    def connect(self):
        return None

    def auth_status(self):
        values = {"user_email": "student@example.edu"}
        if self.user_id is not None:
            values["user_id"] = self.user_id
        return SimpleNamespace(**values)

    def create_project(self, name, **kwargs):
        self.projects.append((name, kwargs))
        return SimpleNamespace(project_id=PROJECT_ID)

    def upload(self, path, **kwargs):
        asset_id = f"asset-{len(self.uploads) + 1}"
        self.record_types[asset_id] = kwargs.get("record_type")
        self.uploads.append(
            {
                "asset_id": asset_id,
                "path": Path(path),
                "kwargs": kwargs,
            }
        )
        return SimpleNamespace(asset_id=asset_id)

    def create_relationship(self, from_asset_id, to_asset_id, **kwargs):
        self.relationships.append((from_asset_id, to_asset_id, kwargs))
        return SimpleNamespace(id=f"rel-{len(self.relationships)}")

    def set_metadata(self, asset_id, **kwargs):
        self.metadata_updates.append((asset_id, kwargs))

    def set_record_type(self, asset_id, *, record_type):
        self.record_types[asset_id] = record_type

    def close(self):
        self.closed = True


def make_tracker(tmp_path: Path, **tracker_kwargs):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    notebook = repo / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")
    client = FakeClient()
    shell = FakeShell()
    tracker = NotebookProvenance(
        notebook,
        client=client,
        shell=shell,
        spool_root=repo / ".dataerai" / "runs",
        **tracker_kwargs,
    ).start()
    return tracker, client, shell


def test_start_creates_notebook_and_run_assets(tmp_path):
    tracker, client, shell = make_tracker(tmp_path)

    assert tracker.owner_type == "user"
    assert tracker.owner_id == USER_ID
    assert tracker.owner_resolution == "authenticated_user"
    assert tracker.notebook_asset_id == "asset-1"
    assert tracker.run_asset_id == "asset-2"
    assert shell.registered[1:] == ("cell", "dataerai")
    assert client.relationships[0][2]["relationship_type"] == "executes_notebook"
    assert client.uploads[1]["kwargs"]["metadata"]["schema"] == SCHEMA


def test_auto_owner_creates_and_reuses_managed_project_for_legacy_sdk(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    notebook = repo / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")
    client = FakeClient(user_id=None)

    first = NotebookProvenance(
        notebook,
        client=client,
        shell=FakeShell(),
        spool_root=repo / ".dataerai" / "runs",
    ).start()
    second = NotebookProvenance(
        notebook,
        client=client,
        shell=FakeShell(),
        spool_root=repo / ".dataerai" / "runs",
    ).start()

    assert first.owner_type == second.owner_type == "project"
    assert first.owner_id == second.owner_id == PROJECT_ID
    assert first.owner_resolution == "managed_project_created"
    assert second.owner_resolution == "managed_project_reused"
    assert len(client.projects) == 1
    state = json.loads((repo / ".dataerai" / "owner-projects.json").read_text())
    assert state["owners"]["student@example.edu"]["project_id"] == PROJECT_ID


def test_email_is_never_accepted_as_an_owner_uuid(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    notebook = repo / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")
    tracker = NotebookProvenance(
        notebook,
        owner_type="user",
        owner_id="student@example.edu",
        client=FakeClient(),
        shell=FakeShell(),
        spool_root=repo / ".dataerai" / "runs",
    )

    with pytest.raises(ValueError, match="must be a Dataerai UUID"):
        tracker.start()


def test_user_owner_requires_new_sdk_or_explicit_uuid(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    notebook = repo / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")
    tracker = NotebookProvenance(
        notebook,
        owner_type="user",
        client=FakeClient(user_id=None),
        shell=FakeShell(),
        spool_root=repo / ".dataerai" / "runs",
    )

    with pytest.raises(RuntimeError, match="email but not the UUID"):
        tracker.start()


def test_record_titles_isolate_tracks_and_repeated_runs(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    student = repo / "4x2" / "labs" / "lesson.ipynb"
    solution = repo / "4x2" / "labs_solns" / "lesson.ipynb"
    student.parent.mkdir(parents=True)
    solution.parent.mkdir(parents=True)
    for notebook in (student, solution):
        notebook.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")

    client = FakeClient()
    first = NotebookProvenance(student, client=client, shell=FakeShell()).start()
    first.record_cell(source="answer = 1", assigned_names=["answer"], user_ns={"answer": 1})
    second = NotebookProvenance(student, client=client, shell=FakeShell()).start()
    second.record_cell(source="answer = 2", assigned_names=["answer"], user_ns={"answer": 2})
    third = NotebookProvenance(solution, client=client, shell=FakeShell()).start()

    source_uploads = [
        upload
        for upload in client.uploads
        if upload["kwargs"]["metadata"]["record_kind"] == "notebook_source"
    ]
    assert source_uploads[0]["kwargs"]["title"] == source_uploads[1]["kwargs"]["title"]
    assert source_uploads[0]["kwargs"]["title"] != source_uploads[2]["kwargs"]["title"]
    assert source_uploads[0]["kwargs"]["metadata"]["tutorial"]["track"] == "analysis"
    assert source_uploads[2]["kwargs"]["metadata"]["tutorial"]["track"] == "analysis"

    cell_titles = [
        upload["kwargs"]["title"]
        for upload in client.uploads
        if upload["kwargs"]["metadata"]["record_kind"] == "cell_execution"
    ]
    assert first.run_id in cell_titles[0]
    assert second.run_id in cell_titles[1]
    assert cell_titles[0] != cell_titles[1]
    assert all(len(title) <= 255 for title in cell_titles)


def test_start_rejects_sdk_without_relationship_support(tmp_path):
    class UploadOnlyClient:
        def connect(self):
            return None

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    notebook = repo / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")
    tracker = NotebookProvenance(
        notebook,
        client=UploadOnlyClient(),
        shell=FakeShell(),
        spool_root=repo / ".dataerai" / "runs",
    )

    with pytest.raises(RuntimeError, match="provenance-capable Dataerai SDK"):
        tracker.start()


def test_start_records_local_and_unresolved_dependencies(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "calibration.pkl").write_bytes(b"calibration")
    notebook = repo / "lesson.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": [
                            "from helper import VALUE\n",
                            "open('calibration.pkl', 'rb')\n",
                            "bitfile = '/missing/qick.bit'\n",
                        ],
                    }
                ],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    client = FakeClient()
    NotebookProvenance(
        notebook,
        client=client,
        shell=FakeShell(),
        spool_root=repo / ".dataerai" / "runs",
    ).start()

    kinds = [upload["kwargs"]["metadata"]["record_kind"] for upload in client.uploads]
    assert kinds.count("notebook_dependency") == 2
    dependency_upload = next(
        upload
        for upload in client.uploads
        if upload["kwargs"]["metadata"]["record_kind"] == "notebook_dependency"
    )
    assert "run" not in dependency_upload["kwargs"]["metadata"]
    assert "tutorial" not in dependency_upload["kwargs"]["metadata"]
    assert "sha256:" in dependency_upload["kwargs"]["title"]
    manifest_upload = next(
        upload
        for upload in client.uploads
        if upload["kwargs"]["metadata"]["record_kind"] == "dependency_manifest"
    )
    manifest = json.loads(manifest_upload["path"].read_text(encoding="utf-8"))
    assert manifest["unresolved_references"][0]["reference"] == "/missing/qick.bit"
    relationships = [item[2]["relationship_type"] for item in client.relationships]
    assert "uses_dependency_manifest" in relationships
    assert relationships.count("uses_dependency") == 2


def test_cells_create_continuous_and_semantic_provenance(tmp_path):
    tracker, client, _ = make_tracker(tmp_path)

    first = tracker.record_cell(
        source="config = {'frequency_mhz': 500}",
        assigned_names=["config"],
        user_ns={"config": {"frequency_mhz": 500}},
        stdout="configured\n",
    )
    second = tracker.record_cell(
        source="iq_data = [1, 2, 3]",
        assigned_names=["iq_data"],
        user_ns={"iq_data": [1, 2, 3]},
        displays=[DisplayOutput(data={"text/plain": "[1, 2, 3]"})],
    )

    relationships = [item[2]["relationship_type"] for item in client.relationships]
    assert first != second
    assert "starts_from_notebook" in relationships
    assert "continues_from" in relationships
    assert "generated_by" in relationships
    assert "acquired_with" in relationships
    cell_records = [
        upload
        for upload in client.uploads
        if upload["kwargs"]["metadata"]["record_kind"] == "cell_execution"
    ]
    assert len(cell_records) == 2
    payload = json.loads(cell_records[1]["path"].read_text(encoding="utf-8"))
    assert payload["execution"]["rich_outputs"][0]["mime"]["text/plain"] == "[1, 2, 3]"


def test_numpy_and_image_outputs_are_individual_records(tmp_path):
    import base64

    import numpy as np

    tracker, client, _ = make_tracker(tmp_path)
    png = base64.b64encode(b"fake-png").decode("ascii")
    tracker.record_cell(
        source="iq = acquire()",
        assigned_names=["iq"],
        user_ns={"iq": np.arange(6).reshape(2, 3)},
        displays=[DisplayOutput(data={"image/png": png})],
    )

    outputs = [
        upload
        for upload in client.uploads
        if upload["kwargs"]["metadata"]["record_kind"] == "cell_output"
    ]
    formats = {item["kwargs"]["metadata"]["output"]["format"] for item in outputs}
    assert formats == {"npz", "png"}
    npz_output = next(item for item in outputs if item["path"].suffix == ".npz")
    png_output = next(item for item in outputs if item["path"].suffix == ".png")
    assert npz_output["kwargs"]["metadata"]["output"]["shape"] == [2, 3]
    assert client.record_types[npz_output["asset_id"]] == "analysis"
    assert client.record_types[png_output["asset_id"]] == "figure"
    analysis_link = next(
        item
        for item in client.relationships
        if item[2]["relationship_type"] == "analysis_of"
    )
    assert analysis_link[2]["analysis_mode"] == "non_destructive"


def test_all_generated_files_keep_the_run_project_and_collection(tmp_path):
    import base64

    import numpy as np

    tracker, client, _ = make_tracker(
        tmp_path,
        owner_type="project",
        owner_id=PROJECT_ID,
        collection_id=COLLECTION_ID,
    )
    png = base64.b64encode(b"generated-plot").decode("ascii")

    # Changing public attributes after startup must not reroute later outputs.
    tracker.owner_type = "user"
    tracker.owner_id = USER_ID
    tracker.collection_id = None
    tracker.record_cell(
        source="iq = acquire()",
        assigned_names=["iq"],
        user_ns={"iq": np.arange(4)},
        displays=[DisplayOutput(data={"image/png": png})],
    )
    tracker.finish()

    generated_formats = {
        upload["kwargs"]["metadata"]["output"]["format"]
        for upload in client.uploads
        if upload["kwargs"]["metadata"]["record_kind"] == "cell_output"
    }
    assert generated_formats == {"npz", "png"}
    for upload in client.uploads:
        kwargs = upload["kwargs"]
        assert kwargs["owner_type"] == "project"
        assert kwargs["owner_id"] == PROJECT_ID
        assert kwargs["collection_id"] == COLLECTION_ID
        assert kwargs["metadata"]["destination"] == {
            "owner_type": "project",
            "owner_id": PROJECT_ID,
            "collection_id": COLLECTION_ID,
        }


def test_record_types_cover_notebook_code_workflow_and_logs(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    simulation = repo / "4x2" / "simulation" / "notebooks" / "lesson.ipynb"
    simulation.parent.mkdir(parents=True)
    simulation.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")
    client = FakeClient()
    tracker = NotebookProvenance(
        simulation,
        client=client,
        shell=FakeShell(),
        spool_root=repo / ".dataerai" / "runs",
    ).start()

    tracker.record_cell(
        source="config = {'frequency': 500}",
        assigned_names=["config"],
        user_ns={"config": {"frequency": 500}},
    )
    tracker.finish()

    typed_by_kind = {}
    for upload in client.uploads:
        kind = upload["kwargs"]["metadata"]["record_kind"]
        typed_by_kind.setdefault(kind, set()).add(client.record_types[upload["asset_id"]])
    assert typed_by_kind["notebook_source"] == {"jupyter_notebook"}
    assert typed_by_kind["notebook_run"] == {"simulation"}
    assert typed_by_kind["cell_output"] == {"protocol_workflow"}
    assert typed_by_kind["cell_execution"] == {"log"}
    assert typed_by_kind["notebook_run_summary"] == {"log"}


def test_secret_named_assignments_are_redacted(tmp_path):
    tracker, client, _ = make_tracker(tmp_path)
    tracker.record_cell(
        source="api_token = get_token()",
        assigned_names=["api_token"],
        user_ns={"api_token": "do-not-store-me"},
    )

    assigned = next(
        upload
        for upload in client.uploads
        if upload["path"].name == "assigned-values.json"
    )
    payload = json.loads(assigned["path"].read_text(encoding="utf-8"))
    assert payload["api_token"] == "<redacted>"
    assert "do-not-store-me" not in assigned["path"].read_text(encoding="utf-8")


def test_finish_uploads_summary_and_updates_run(tmp_path):
    tracker, client, _ = make_tracker(tmp_path)
    tracker.record_cell(
        source="answer = 42", assigned_names=["answer"], user_ns={"answer": 42}
    )

    summary_id = tracker.finish()

    assert summary_id is not None
    assert tracker.running is False
    assert client.metadata_updates[0][0] == tracker.run_asset_id
    assert (
        client.metadata_updates[0][1]["metadata"]["completion"]["status"] == "completed"
    )
    relationships = [item[2]["relationship_type"] for item in client.relationships]
    assert "summarizes" in relationships


def test_subscript_assignment_recaptures_mutated_configuration(tmp_path):
    tracker, _, shell = make_tracker(tmp_path)

    names = tracker._assigned_names("config['frequency'] = 501")

    assert names == ["config"]


def test_cell_error_is_recorded_and_propagated(tmp_path):
    class ErrorShell(FakeShell):
        display_pub = None

        def run_cell(self, source, store_history=False):
            del source, store_history
            return SimpleNamespace(
                success=False,
                result=None,
                error_before_exec=None,
                error_in_exec=ValueError("bad experiment cell"),
            )

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    notebook = repo / "lesson.ipynb"
    notebook.write_text('{"cells": [], "metadata": {}}', encoding="utf-8")
    client = FakeClient()
    tracker = NotebookProvenance(
        notebook,
        client=client,
        shell=ErrorShell(),
        spool_root=repo / ".dataerai" / "runs",
    ).start()

    with pytest.raises(ValueError, match="bad experiment cell"):
        tracker._cell_magic("", "raise ValueError('bad experiment cell')")

    cell_upload = next(
        upload
        for upload in client.uploads
        if upload["kwargs"]["metadata"]["record_kind"] == "cell_execution"
    )
    assert cell_upload["kwargs"]["metadata"]["execution"]["status"] == "failed"


def test_nested_result_arrays_preserve_shapes_values_and_container_structure(tmp_path):
    import numpy as np
    tracker, client, _ = make_tracker(tmp_path)
    values = {'sample': {'curves': [np.arange(12).reshape(3, 4), 'units: nm']},
              'threshold': 200, 'api_token': np.array([123])}
    tracker.record_cell(source='results = computed_results', assigned_names=['results'], user_ns={'results': values})
    archive = next(item['path'] for item in client.uploads if item['path'].name == 'results.npz')
    with np.load(archive, allow_pickle=False) as stored:
        assert set(stored.files) == {'array_0000', '__structure__'}
        np.testing.assert_array_equal(stored['array_0000'], values['sample']['curves'][0])
        structure = json.loads(str(stored['__structure__']))
        assert structure['sample']['curves']['items'][1] == 'units: nm'
        assert structure['threshold'] == 200
        assert structure['api_token'] == '<redacted>'


def test_large_rich_display_is_not_truncated(tmp_path):
    tracker, client, _ = make_tracker(tmp_path)
    points = list(range(20001))
    tracker.record_cell(source='display(chart)', displays=[DisplayOutput(data={'application/json': {'points': points}})])
    cell = next(item['path'] for item in client.uploads if item['path'].name == 'cell-execution.json')
    data = json.loads(cell.read_text())
    assert data['execution']['rich_outputs'][0]['mime']['application/json']['points'] == points
