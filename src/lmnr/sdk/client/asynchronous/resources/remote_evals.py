"""Remote evaluations resource for interacting with Laminar remote execution API."""

import uuid
from typing import Dict, Any, Optional

from lmnr.sdk.client.asynchronous.resources.base import BaseAsyncResource


class AsyncRemoteEvals(BaseAsyncResource):
    """Resource for interacting with Laminar remote evaluations API."""

    async def upload_bundle(self, bundle_data: bytes, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Upload evaluation bundle to Laminar servers.

        Args:
            bundle_data (bytes): The bundle zip file data
            metadata (Dict[str, Any], optional): Additional metadata for the bundle

        Returns:
            Dict[str, Any]: Response containing bundle_id and upload status
        """
        # For file uploads, we need to use multipart/form-data instead of JSON
        files = {"bundle": ("bundle.zip", bundle_data, "application/zip")}
        data = {"metadata": str(metadata) if metadata else "{}"}
        
        response = await self._client.post(
            self._base_url + "/v1/remote-evals/bundles",
            files=files,
            data=data,
            headers={
                "Authorization": "Bearer " + self._project_api_key,
                "Accept": "application/json",
            },
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error uploading bundle: {response.text}")
        
        return response.json()

    async def start_remote_evaluation(
        self,
        bundle_id: str,
        dataset_name: str,
        evaluation_name: str | None = None,
        group_name: str | None = None,
        concurrency_limit: int = 50,
        executor_func_name: str = "run_agent",
        evaluators_config: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Start remote evaluation execution.

        Args:
            bundle_id (str): ID of the uploaded bundle
            dataset_name (str): Name of the LaminarDataset to evaluate
            evaluation_name (str | None, optional): Name for the evaluation
            group_name (str | None, optional): Group name for the evaluation
            concurrency_limit (int, optional): Maximum concurrent Modal function calls
            executor_func_name (str, optional): Name of the executor function in the bundle
            evaluators_config (Dict[str, str], optional): Mapping of evaluator names to function names

        Returns:
            Dict[str, Any]: Response containing evaluation_id and execution status
        """
        response = await self._client.post(
            self._base_url + "/v1/remote-evals/execute",
            json={
                "bundleId": bundle_id,
                "datasetName": dataset_name,
                "evaluationName": evaluation_name,
                "groupName": group_name,
                "concurrencyLimit": concurrency_limit,
                "executorFuncName": executor_func_name,
                "evaluatorsConfig": evaluators_config or {}
            },
            headers=self._headers(),
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error starting remote evaluation: {response.text}")
        
        return response.json()

    async def get_evaluation_status(self, evaluation_id: str) -> Dict[str, Any]:
        """Get the status of a remote evaluation.

        Args:
            evaluation_id (str): ID of the evaluation

        Returns:
            Dict[str, Any]: Status information including progress and results
        """
        response = await self._client.get(
            self._base_url + f"/v1/remote-evals/status/{evaluation_id}",
            headers=self._headers(),
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error getting evaluation status: {response.text}")
        
        return response.json()

    async def list_bundles(self) -> Dict[str, Any]:
        """List all uploaded bundles for the current project.

        Returns:
            Dict[str, Any]: List of bundle information
        """
        response = await self._client.get(
            self._base_url + "/v1/remote-evals/bundles",
            headers=self._headers(),
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error listing bundles: {response.text}")
        
        return response.json()

    async def delete_bundle(self, bundle_id: str) -> Dict[str, Any]:
        """Delete an uploaded bundle.

        Args:
            bundle_id (str): ID of the bundle to delete

        Returns:
            Dict[str, Any]: Deletion confirmation
        """
        response = await self._client.delete(
            self._base_url + f"/v1/remote-evals/bundles/{bundle_id}",
            headers=self._headers(),
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error deleting bundle: {response.text}")
        
        return response.json()

    async def get_bundle_info(self, bundle_id: str) -> Dict[str, Any]:
        """Get information about a specific bundle.

        Args:
            bundle_id (str): ID of the bundle

        Returns:
            Dict[str, Any]: Bundle information including metadata
        """
        response = await self._client.get(
            self._base_url + f"/v1/remote-evals/bundles/{bundle_id}",
            headers=self._headers(),
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error getting bundle info: {response.text}")
        
        return response.json() 