import json
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

import yaml

from pingpoint.cli import app, _get_repo_from_git

runner = CliRunner()


class TestVersionCommand:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "pingpoint v0.1.0" in result.stdout


class TestProfileCommand:
    @patch("pingpoint.cli.get_profile")
    def test_profile_output(self, mock_profile, low_profile):
        mock_profile.return_value = low_profile
        result = runner.invoke(app, ["profile"])
        assert result.exit_code == 0
        assert "Platform:" in result.stdout
        assert "CPU:" in result.stdout
        assert "Cores:" in result.stdout
        assert "RAM:" in result.stdout
        assert "Capability:" in result.stdout


class TestAssignCommand:
    @patch("pingpoint.cli.get_profile")
    def test_assign_ollama_not_running(self, mock_profile, low_profile):
        low_profile.ollama_running = False
        mock_profile.return_value = low_profile
        result = runner.invoke(app, ["assign"])
        assert result.exit_code == 1
        assert "No running backend detected" in result.stdout

    @patch("pingpoint.cli.get_profile")
    def test_assign_no_models(self, mock_profile, low_profile):
        low_profile.ollama_running = True
        low_profile.ollama_models = []
        mock_profile.return_value = low_profile
        result = runner.invoke(app, ["assign"])
        assert result.exit_code == 1
        assert "No models found" in result.stdout


class TestValidateCommand:
    def test_validate_unknown_task(self):
        result = runner.invoke(app, ["validate", "nonexistent"])
        assert result.exit_code == 1
        assert "FAIL" in result.stdout or "Error" in result.stdout or "None" in result.stdout or "invalid" in result.stdout


class TestVerifyCommand:
    def test_verify_unknown_task(self):
        result = runner.invoke(app, ["verify", "nonexistent"])
        assert result.exit_code == 0
        assert "VALID" in result.stdout or "TAMPERED" in result.stdout


class TestShowCommand:
    def test_show_unknown_task(self):
        result = runner.invoke(app, ["show", "nonexistent"])
        assert result.exit_code == 1
        assert "No solutions found" in result.stdout


class TestRunCommand:
    @patch("pingpoint.cli.get_profile")
    def test_run_ollama_not_running(self, mock_profile, low_profile):
        low_profile.ollama_running = False
        mock_profile.return_value = low_profile
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
        assert "No running backend detected" in result.stdout

    @patch("pingpoint.cli.get_profile")
    def test_run_no_models(self, mock_profile, low_profile):
        low_profile.ollama_running = True
        low_profile.ollama_models = []
        mock_profile.return_value = low_profile
        result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
        assert "No models found" in result.stdout


class TestGetRepoFromGit:
    @patch("pingpoint.cli.subprocess.run")
    def test_https_url(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/user/repo.git\n",
        )
        assert _get_repo_from_git() == "user/repo"

    @patch("pingpoint.cli.subprocess.run")
    def test_ssh_url(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="git@github.com:user/repo.git\n",
        )
        assert _get_repo_from_git() == "user/repo"

    @patch("pingpoint.cli.subprocess.run")
    def test_ssh_no_suffix(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="git@github.com:user/repo\n",
        )
        assert _get_repo_from_git() == "user/repo"

    @patch("pingpoint.cli.subprocess.run")
    def test_not_github(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://gitlab.com/user/repo.git\n",
        )
        assert _get_repo_from_git() is None

    @patch("pingpoint.cli.subprocess.run")
    def test_git_not_available(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        assert _get_repo_from_git() is None


class TestFetchIssueCommand:
    @patch("pingpoint.cli.urllib.request.urlopen")
    @patch("pingpoint.cli._get_repo_from_git")
    def test_fetch_issue_with_auto_repo(self, mock_repo, mock_urlopen):
        mock_repo.return_value = "user/repo"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "title": "Fix the login bug",
            "body": "Users cannot log in when the password contains special characters.",
            "labels": [{"name": "bug"}, {"name": "security"}],
            "html_url": "https://github.com/user/repo/issues/42",
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = runner.invoke(app, ["fetch-issue", "42"])
        assert result.exit_code == 0
        assert "Task created:" in result.stdout
        assert "Fix the login bug" in result.stdout
        assert "bug" in result.stdout

        yaml_path = Path("tasks") / "issue-42.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        assert data["title"] == "Fix the login bug"
        assert data["issue_number"] == 42
        assert "bug" in data["tags"]
        yaml_path.unlink()

    @patch("pingpoint.cli.urllib.request.urlopen")
    def test_fetch_issue_pr_detected(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "title": "Some PR",
            "pull_request": {"url": "https://api.github.com/repos/user/repo/pulls/1"},
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = runner.invoke(app, ["fetch-issue", "1", "--repo", "user/repo"])
        assert result.exit_code == 1
        assert "pull request" in result.stdout.lower()

    @patch("pingpoint.cli.urllib.request.urlopen")
    def test_fetch_issue_404(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            "https://api.github.com/repos/user/repo/issues/999",
            404, "Not Found", {}, None,
        )
        result = runner.invoke(app, ["fetch-issue", "999", "--repo", "user/repo"])
        assert result.exit_code == 1
        assert "404" in result.stdout or "not found" in result.stdout.lower()

    @patch("pingpoint.cli._get_repo_from_git")
    def test_fetch_issue_no_repo_detected(self, mock_repo):
        mock_repo.return_value = None
        result = runner.invoke(app, ["fetch-issue", "1"])
        assert result.exit_code == 1
        assert "Could not detect repo" in result.stdout
