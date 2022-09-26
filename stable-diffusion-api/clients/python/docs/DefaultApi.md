# swagger_client.DefaultApi

All URIs are relative to *https://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**txt2img**](DefaultApi.md#txt2img) | **POST** /api/txt2img | Generate an image using a text prompt


# **txt2img**
> Txt2ImgOutput txt2img(body)

Generate an image using a text prompt

Generate an image using a text prompt

### Example
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.DefaultApi()
body = swagger_client.Txt2ImgInput() # Txt2ImgInput | 

try:
    # Generate an image using a text prompt
    api_response = api_instance.txt2img(body)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling DefaultApi->txt2img: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | [**Txt2ImgInput**](Txt2ImgInput.md)|  | 

### Return type

[**Txt2ImgOutput**](Txt2ImgOutput.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: Not defined

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

