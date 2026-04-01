/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "aeongui/AeonGUI.hpp"
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"

using Microsoft::WRL::ComPtr;

namespace
{
    constexpr UINT FrameCount = 2;

    void ThrowIfFailed ( HRESULT hr, const char* message )
    {
        if ( FAILED ( hr ) )
        {
            throw std::runtime_error ( message );
        }
    }

    ComPtr<IDXGIAdapter1> GetHardwareAdapter ( IDXGIFactory6* factory )
    {
        ComPtr<IDXGIAdapter1> adapter;
        for ( UINT index = 0;
              factory->EnumAdapterByGpuPreference ( index,
                  DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                  IID_PPV_ARGS ( adapter.ReleaseAndGetAddressOf() ) ) != DXGI_ERROR_NOT_FOUND;
              ++index )
        {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1 ( &desc );
            if ( desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE )
            {
                continue;
            }

            if ( SUCCEEDED ( D3D12CreateDevice ( adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                                 __uuidof ( ID3D12Device ), nullptr ) ) )
            {
                return adapter;
            }
        }

        return nullptr;
    }

    const char* kVertexShader = R"(
struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

VSOut main(uint vertexId : SV_VertexID)
{
    float2 pos[4] = {
        float2(-1.0,  1.0),
        float2( 1.0,  1.0),
        float2(-1.0, -1.0),
        float2( 1.0, -1.0)
    };

    float2 uv[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 1.0)
    };

    VSOut outv;
    outv.pos = float4(pos[vertexId], 0.0, 1.0);
    outv.uv = uv[vertexId];
    return outv;
}
)";

    const char* kPixelShader = R"(
Texture2D screenTexture : register(t0);
SamplerState screenSampler : register(s0);

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    return screenTexture.Sample(screenSampler, uv);
}
)";
} // namespace

class D3D12Window
{
public:
    D3D12Window(HINSTANCE hInstance, const char* filename, LONG width, LONG height)
        : mWidth(static_cast<uint32_t>(width))
        , mHeight(static_cast<uint32_t>(height))
        , mAeonWindow(mWidth, mHeight)
    {
        if (filename)
        {
            mAeonWindow.location() = filename;
        }

        InitializeWindow(hInstance, width, height);
        InitializeD3D12();
        ShowWindow(mHWnd, SW_SHOW);
    }

    ~D3D12Window()
    {
        try
        {
            WaitForGpu();
            if (mFenceEvent)
            {
                CloseHandle(mFenceEvent);
                mFenceEvent = nullptr;
            }
            if (mHWnd)
            {
                DestroyWindow(mHWnd);
                mHWnd = nullptr;
            }
        }
        catch (...)
        {
        }
    }

    void RenderLoop()
    {
        PopulateCommandList();

        ID3D12CommandList* commandLists[] = { mCommandList.Get() };
        mCommandQueue->ExecuteCommandLists(1, commandLists);

        ThrowIfFailed(mSwapChain->Present(1, 0), "Failed to present frame.");
        MoveToNextFrame();
    }

    static void Register(HINSTANCE hInstance)
    {
        WNDCLASSEX wcex{};
        wcex.cbSize = sizeof(WNDCLASSEX);
        wcex.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
        wcex.lpfnWndProc = D3D12Window::WindowProc;
        wcex.cbWndExtra = sizeof(D3D12Window*);
        wcex.hInstance = hInstance;
        wcex.hIcon = LoadIcon(NULL, IDI_WINLOGO);
        wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
        wcex.lpszClassName = "AeonGUIDirect3D12";
        atom = RegisterClassEx(&wcex);
    }

    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
    {
        auto* self = reinterpret_cast<D3D12Window*>(GetWindowLongPtr(hwnd, 0));

        switch (uMsg)
        {
        case WM_CLOSE:
            PostQuitMessage(0);
            return 0;
        case WM_SIZE:
            if (self)
            {
                self->OnResize(LOWORD(lParam), HIWORD(lParam));
            }
            return 0;
        case WM_PAINT:
        {
            PAINTSTRUCT ps{};
            BeginPaint(hwnd, &ps);
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_MOUSEMOVE:
            if (self)
            {
                self->mAeonWindow.HandleMouseMove(
                    static_cast<double>(GET_X_LPARAM(lParam)),
                    static_cast<double>(GET_Y_LPARAM(lParam)));
            }
            return 0;
        case WM_LBUTTONDOWN:
            if (self)
            {
                self->mAeonWindow.HandleMouseDown(
                    static_cast<double>(GET_X_LPARAM(lParam)),
                    static_cast<double>(GET_Y_LPARAM(lParam)), 0);
            }
            return 0;
        case WM_LBUTTONUP:
            if (self)
            {
                self->mAeonWindow.HandleMouseUp(
                    static_cast<double>(GET_X_LPARAM(lParam)),
                    static_cast<double>(GET_Y_LPARAM(lParam)), 0);
            }
            return 0;
        case WM_MBUTTONDOWN:
            if (self)
            {
                self->mAeonWindow.HandleMouseDown(
                    static_cast<double>(GET_X_LPARAM(lParam)),
                    static_cast<double>(GET_Y_LPARAM(lParam)), 1);
            }
            return 0;
        case WM_MBUTTONUP:
            if (self)
            {
                self->mAeonWindow.HandleMouseUp(
                    static_cast<double>(GET_X_LPARAM(lParam)),
                    static_cast<double>(GET_Y_LPARAM(lParam)), 1);
            }
            return 0;
        case WM_RBUTTONDOWN:
            if (self)
            {
                self->mAeonWindow.HandleMouseDown(
                    static_cast<double>(GET_X_LPARAM(lParam)),
                    static_cast<double>(GET_Y_LPARAM(lParam)), 2);
            }
            return 0;
        case WM_RBUTTONUP:
            if (self)
            {
                self->mAeonWindow.HandleMouseUp(
                    static_cast<double>(GET_X_LPARAM(lParam)),
                    static_cast<double>(GET_Y_LPARAM(lParam)), 2);
            }
            return 0;
        case WM_MOUSEWHEEL:
            if (self)
            {
                POINT pt = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
                ScreenToClient(hwnd, &pt);
                self->mAeonWindow.HandleWheel(
                    static_cast<double>(pt.x),
                    static_cast<double>(pt.y),
                    0.0,
                    static_cast<double>(-GET_WHEEL_DELTA_WPARAM(wParam)));
            }
            return 0;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
    }

private:
    void InitializeWindow(HINSTANCE hInstance, LONG width, LONG height)
    {
        if (atom == 0)
        {
            Register(hInstance);
        }

        RECT rect = { 0, 0, width, height };
        AdjustWindowRectEx(&rect,
                           WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                           FALSE,
                           WS_EX_APPWINDOW | WS_EX_WINDOWEDGE);

        mHWnd = CreateWindowEx(
                    WS_EX_APPWINDOW | WS_EX_WINDOWEDGE,
                    "AeonGUIDirect3D12",
                    "AeonGUI - Direct3D12",
                    WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                    CW_USEDEFAULT,
                    CW_USEDEFAULT,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                    NULL,
                    NULL,
                    hInstance,
                    this);

        if (!mHWnd)
        {
            throw std::runtime_error("Failed to create window.");
        }

        SetWindowLongPtr(mHWnd, 0, reinterpret_cast<LONG_PTR>(this));
    }

    void InitializeD3D12()
    {
        UINT dxgiFactoryFlags = 0;

#ifdef _DEBUG
        {
            ComPtr<ID3D12Debug> debugController;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
            {
                debugController->EnableDebugLayer();
                dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            }
        }
#endif

        ComPtr<IDXGIFactory6> factory;
        ThrowIfFailed(
            CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)),
            "Failed to create DXGI factory.");

        ComPtr<IDXGIAdapter1> adapter = GetHardwareAdapter(factory.Get());
        if (adapter)
        {
            ThrowIfFailed(
                D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&mDevice)),
                "Failed to create D3D12 device.");
        }
        else
        {
            ComPtr<IDXGIAdapter> warpAdapter;
            ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)),
                          "Failed to enumerate WARP adapter.");
            ThrowIfFailed(
                D3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&mDevice)),
                "Failed to create D3D12 WARP device.");
        }

        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ThrowIfFailed(mDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&mCommandQueue)),
                      "Failed to create command queue.");

        DXGI_SWAP_CHAIN_DESC1 swapDesc{};
        swapDesc.BufferCount = FrameCount;
        swapDesc.Width = mWidth;
        swapDesc.Height = mHeight;
        swapDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapDesc.SampleDesc.Count = 1;

        ComPtr<IDXGISwapChain1> swapChain;
        ThrowIfFailed(
            factory->CreateSwapChainForHwnd(
                mCommandQueue.Get(),
                mHWnd,
                &swapDesc,
                nullptr,
                nullptr,
                &swapChain),
            "Failed to create swap chain.");

        ThrowIfFailed(factory->MakeWindowAssociation(mHWnd, DXGI_MWA_NO_ALT_ENTER),
                      "Failed to set window association.");
        ThrowIfFailed(swapChain.As(&mSwapChain), "Failed to get swap chain 3 interface.");
        mFrameIndex = mSwapChain->GetCurrentBackBufferIndex();

        D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
        rtvHeapDesc.NumDescriptors = FrameCount;
        rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        ThrowIfFailed(mDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&mRtvHeap)),
                      "Failed to create RTV heap.");
        mRtvDescriptorSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc{};
        srvHeapDesc.NumDescriptors = 1;
        srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(mDevice->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&mSrvHeap)),
                      "Failed to create SRV heap.");

        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < FrameCount; ++i)
        {
            ThrowIfFailed(
                mSwapChain->GetBuffer(i, IID_PPV_ARGS(&mRenderTargets[i])),
                "Failed to get render target from swap chain.");
            mDevice->CreateRenderTargetView(mRenderTargets[i].Get(), nullptr, rtvHandle);
            rtvHandle.ptr += mRtvDescriptorSize;

            ThrowIfFailed(
                mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_PPV_ARGS(&mCommandAllocators[i])),
                "Failed to create command allocator.");
        }

        ThrowIfFailed(
            mDevice->CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                mCommandAllocators[mFrameIndex].Get(),
                nullptr,
                IID_PPV_ARGS(&mCommandList)),
            "Failed to create command list.");
        ThrowIfFailed(mCommandList->Close(), "Failed to close initial command list.");

        ThrowIfFailed(mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFence)),
                      "Failed to create fence.");
        mFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (!mFenceEvent)
        {
            throw std::runtime_error("Failed to create fence event.");
        }

        mFenceValues.fill(0);

        CreatePipelineState();
        CreateOrResizeTexture();

        mViewport = {
                        0.0f,
                        0.0f,
                        static_cast<float>(mWidth),
                        static_cast<float>(mHeight),
                        0.0f,
                        1.0f
                    };
        mScissorRect = { 0, 0, static_cast<LONG>(mWidth), static_cast<LONG>(mHeight) };
    }

    void CreatePipelineState()
    {
        ComPtr<ID3DBlob> vertexShader;
        ComPtr<ID3DBlob> pixelShader;
        ComPtr<ID3DBlob> errorBlob;

        UINT shaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
        shaderFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

        HRESULT hr = D3DCompile(kVertexShader, std::strlen(kVertexShader), nullptr,
                                nullptr, nullptr, "main", "vs_5_0", shaderFlags, 0,
                                &vertexShader, &errorBlob);
        if (FAILED(hr))
        {
            if (errorBlob)
            {
                std::cerr << static_cast<const char*>(errorBlob->GetBufferPointer()) << std::endl;
            }
            ThrowIfFailed(hr, "Failed to compile D3D12 vertex shader.");
        }

        hr = D3DCompile(kPixelShader, std::strlen(kPixelShader), nullptr,
                        nullptr, nullptr, "main", "ps_5_0", shaderFlags, 0,
                        &pixelShader, &errorBlob);
        if (FAILED(hr))
        {
            if (errorBlob)
            {
                std::cerr << static_cast<const char*>(errorBlob->GetBufferPointer()) << std::endl;
            }
            ThrowIfFailed(hr, "Failed to compile D3D12 pixel shader.");
        }

        D3D12_DESCRIPTOR_RANGE1 range{};
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = 0;
        range.RegisterSpace = 0;
        range.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;

        D3D12_ROOT_PARAMETER1 rootParameter{};
        rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        rootParameter.DescriptorTable.NumDescriptorRanges = 1;
        rootParameter.DescriptorTable.pDescriptorRanges = &range;

        D3D12_STATIC_SAMPLER_DESC sampler{};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        sampler.ShaderRegister = 0;

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc{};
        rootSignatureDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        rootSignatureDesc.Desc_1_1.NumParameters = 1;
        rootSignatureDesc.Desc_1_1.pParameters = &rootParameter;
        rootSignatureDesc.Desc_1_1.NumStaticSamplers = 1;
        rootSignatureDesc.Desc_1_1.pStaticSamplers = &sampler;
        rootSignatureDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ComPtr<ID3DBlob> serializedRootSig;
        ThrowIfFailed(
            D3D12SerializeVersionedRootSignature(&rootSignatureDesc,
                    &serializedRootSig,
                    &errorBlob),
            "Failed to serialize root signature.");

        ThrowIfFailed(
            mDevice->CreateRootSignature(0,
                                         serializedRootSig->GetBufferPointer(),
                                         serializedRootSig->GetBufferSize(),
                                         IID_PPV_ARGS(&mRootSignature)),
            "Failed to create root signature.");

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = mRootSignature.Get();
        psoDesc.VS = { vertexShader->GetBufferPointer(), vertexShader->GetBufferSize() };
        psoDesc.PS = { pixelShader->GetBufferPointer(), pixelShader->GetBufferSize() };

        D3D12_BLEND_DESC blendDesc{};
        blendDesc.AlphaToCoverageEnable = FALSE;
        blendDesc.IndependentBlendEnable = FALSE;
        const D3D12_RENDER_TARGET_BLEND_DESC defaultBlendTarget{
            TRUE,
            FALSE,
            D3D12_BLEND_SRC_ALPHA,
            D3D12_BLEND_INV_SRC_ALPHA,
            D3D12_BLEND_OP_ADD,
            D3D12_BLEND_ONE,
            D3D12_BLEND_INV_SRC_ALPHA,
            D3D12_BLEND_OP_ADD,
            D3D12_LOGIC_OP_NOOP,
            D3D12_COLOR_WRITE_ENABLE_ALL
        };
        for (auto& renderTargetBlend : blendDesc.RenderTarget)
        {
            renderTargetBlend = defaultBlendTarget;
        }

        D3D12_RASTERIZER_DESC rasterizerDesc{};
        rasterizerDesc.FillMode = D3D12_FILL_MODE_SOLID;
        rasterizerDesc.CullMode = D3D12_CULL_MODE_BACK;
        rasterizerDesc.FrontCounterClockwise = FALSE;
        rasterizerDesc.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
        rasterizerDesc.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
        rasterizerDesc.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
        rasterizerDesc.DepthClipEnable = TRUE;
        rasterizerDesc.MultisampleEnable = FALSE;
        rasterizerDesc.AntialiasedLineEnable = FALSE;
        rasterizerDesc.ForcedSampleCount = 0;
        rasterizerDesc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

        psoDesc.BlendState = blendDesc;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.RasterizerState = rasterizerDesc;
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.DepthStencilState.StencilEnable = FALSE;
        psoDesc.InputLayout = { nullptr, 0 };
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
        psoDesc.SampleDesc.Count = 1;

        ThrowIfFailed(
            mDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPipelineState)),
            "Failed to create pipeline state.");
    }

    void CreateOrResizeTexture()
    {
        const UINT texWidth = static_cast<UINT>(mAeonWindow.GetWidth());
        const UINT texHeight = static_cast<UINT>(mAeonWindow.GetHeight());

        mTexture.Reset();
        mTextureUpload.Reset();

        D3D12_RESOURCE_DESC texDesc{};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Width = texWidth;
        texDesc.Height = texHeight;
        texDesc.DepthOrArraySize = 1;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

        D3D12_HEAP_PROPERTIES defaultHeap{};
        defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

        ThrowIfFailed(
            mDevice->CreateCommittedResource(
                &defaultHeap,
                D3D12_HEAP_FLAG_NONE,
                &texDesc,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                nullptr,
                IID_PPV_ARGS(&mTexture)),
            "Failed to create texture resource.");

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;

        mDevice->CreateShaderResourceView(
            mTexture.Get(),
            &srvDesc,
            mSrvHeap->GetCPUDescriptorHandleForHeapStart());

        mDevice->GetCopyableFootprints(&texDesc, 0, 1, 0, &mTextureFootprint, nullptr, nullptr, &mUploadBufferSize);

        D3D12_HEAP_PROPERTIES uploadHeap{};
        uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC uploadDesc{};
        uploadDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        uploadDesc.Width = mUploadBufferSize;
        uploadDesc.Height = 1;
        uploadDesc.DepthOrArraySize = 1;
        uploadDesc.MipLevels = 1;
        uploadDesc.SampleDesc.Count = 1;
        uploadDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        ThrowIfFailed(
            mDevice->CreateCommittedResource(
                &uploadHeap,
                D3D12_HEAP_FLAG_NONE,
                &uploadDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&mTextureUpload)),
            "Failed to create upload texture buffer.");
    }

    void UploadAeonTexture()
    {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(mAeonWindow.GetPixels());
        const UINT srcRowPitch = static_cast<UINT>(mAeonWindow.GetStride());
        const UINT rowPitch = mTextureFootprint.Footprint.RowPitch;
        const UINT rowCount = mTextureFootprint.Footprint.Height;
        const UINT copyBytes = (srcRowPitch < rowPitch) ? srcRowPitch : rowPitch;

        uint8_t* mapped = nullptr;
        D3D12_RANGE readRange{ 0, 0 };
        ThrowIfFailed(mTextureUpload->Map(0, &readRange, reinterpret_cast<void**>(&mapped)),
                      "Failed to map upload texture.");

        for (UINT y = 0; y < rowCount; ++y)
        {
            std::memcpy(mapped + (rowPitch * y), src + (srcRowPitch * y), copyBytes);
        }

        D3D12_RANGE writeRange{ 0, mUploadBufferSize };
        mTextureUpload->Unmap(0, &writeRange);
    }

    void PopulateCommandList()
    {
        ThrowIfFailed(
            mCommandAllocators[mFrameIndex]->Reset(),
            "Failed to reset command allocator.");

        ThrowIfFailed(
            mCommandList->Reset(mCommandAllocators[mFrameIndex].Get(), mPipelineState.Get()),
            "Failed to reset command list.");

        mAeonWindow.Update ( 1.0 / 60.0 );
        if ( mAeonWindow.Draw() )
        {
            UploadAeonTexture();
        }

        D3D12_RESOURCE_BARRIER barriers[2]{};
        barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[0].Transition.pResource = mTexture.Get();
        barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;

        barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[1].Transition.pResource = mRenderTargets[mFrameIndex].Get();
        barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;

        mCommandList->ResourceBarrier(2, barriers);

        D3D12_TEXTURE_COPY_LOCATION dstLocation{};
        dstLocation.pResource = mTexture.Get();
        dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLocation.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION srcLocation{};
        srcLocation.pResource = mTextureUpload.Get();
        srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLocation.PlacedFootprint = mTextureFootprint;

        mCommandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);

        D3D12_RESOURCE_BARRIER textureToShader{};
        textureToShader.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        textureToShader.Transition.pResource = mTexture.Get();
        textureToShader.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        textureToShader.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        textureToShader.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        mCommandList->ResourceBarrier(1, &textureToShader);

        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
        rtvHandle.ptr += static_cast<SIZE_T>(mFrameIndex) * static_cast<SIZE_T>(mRtvDescriptorSize);

        mCommandList->RSSetViewports(1, &mViewport);
        mCommandList->RSSetScissorRects(1, &mScissorRect);
        mCommandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

        constexpr float clearColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        mCommandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

        ID3D12DescriptorHeap* heaps[] = { mSrvHeap.Get() };
        mCommandList->SetDescriptorHeaps(1, heaps);
        mCommandList->SetGraphicsRootSignature(mRootSignature.Get());
        mCommandList->SetGraphicsRootDescriptorTable(0, mSrvHeap->GetGPUDescriptorHandleForHeapStart());
        mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        mCommandList->DrawInstanced(4, 1, 0, 0);

        D3D12_RESOURCE_BARRIER toPresent{};
        toPresent.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toPresent.Transition.pResource = mRenderTargets[mFrameIndex].Get();
        toPresent.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        toPresent.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        toPresent.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        mCommandList->ResourceBarrier(1, &toPresent);

        ThrowIfFailed(mCommandList->Close(), "Failed to close command list.");
    }

    void WaitForGpu()
    {
        const UINT64 value = ++mFenceValues[mFrameIndex];
        ThrowIfFailed(
            mCommandQueue->Signal(mFence.Get(), value),
            "Failed to signal fence.");

        ThrowIfFailed(
            mFence->SetEventOnCompletion(value, mFenceEvent),
            "Failed to set fence completion event.");
        WaitForSingleObject(mFenceEvent, INFINITE);
    }

    void MoveToNextFrame()
    {
        const UINT64 currentFence = ++mFenceValues[mFrameIndex];
        ThrowIfFailed(
            mCommandQueue->Signal(mFence.Get(), currentFence),
            "Failed to signal fence for frame transition.");

        mFrameIndex = mSwapChain->GetCurrentBackBufferIndex();

        if (mFence->GetCompletedValue() < mFenceValues[mFrameIndex])
        {
            ThrowIfFailed(
                mFence->SetEventOnCompletion(mFenceValues[mFrameIndex], mFenceEvent),
                "Failed to set frame fence completion event.");
            WaitForSingleObject(mFenceEvent, INFINITE);
        }

        mFenceValues[mFrameIndex] = currentFence;
    }

    void OnResize(WORD newWidth, WORD newHeight)
    {
        if (newWidth == 0 || newHeight == 0)
        {
            return;
        }

        mWidth = static_cast<uint32_t>(newWidth);
        mHeight = static_cast<uint32_t>(newHeight);

        WaitForGpu();

        for (UINT i = 0; i < FrameCount; ++i)
        {
            mRenderTargets[i].Reset();
        }

        DXGI_SWAP_CHAIN_DESC swapDesc{};
        ThrowIfFailed(mSwapChain->GetDesc(&swapDesc), "Failed to get swap chain desc.");
        ThrowIfFailed(
            mSwapChain->ResizeBuffers(FrameCount, mWidth, mHeight, swapDesc.BufferDesc.Format, swapDesc.Flags),
            "Failed to resize swap chain buffers.");

        mFrameIndex = mSwapChain->GetCurrentBackBufferIndex();

        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < FrameCount; ++i)
        {
            ThrowIfFailed(mSwapChain->GetBuffer(i, IID_PPV_ARGS(&mRenderTargets[i])),
                          "Failed to reacquire swap chain render target.");
            mDevice->CreateRenderTargetView(mRenderTargets[i].Get(), nullptr, rtvHandle);
            rtvHandle.ptr += mRtvDescriptorSize;
        }

        mViewport.Width = static_cast<float>(mWidth);
        mViewport.Height = static_cast<float>(mHeight);
        mScissorRect.right = static_cast<LONG>(mWidth);
        mScissorRect.bottom = static_cast<LONG>(mHeight);

        mAeonWindow.ResizeViewport(static_cast<size_t>(mWidth), static_cast<size_t>(mHeight));
        CreateOrResizeTexture();
    }

private:
    static ATOM atom;

    HWND mHWnd{};
    uint32_t mWidth{};
    uint32_t mHeight{};

    AeonGUI::DOM::Window mAeonWindow;

    ComPtr<ID3D12Device> mDevice;
    ComPtr<ID3D12CommandQueue> mCommandQueue;
    ComPtr<IDXGISwapChain3> mSwapChain;
    ComPtr<ID3D12DescriptorHeap> mRtvHeap;
    ComPtr<ID3D12DescriptorHeap> mSrvHeap;
    std::array<ComPtr<ID3D12Resource>, FrameCount> mRenderTargets;
    std::array<ComPtr<ID3D12CommandAllocator>, FrameCount> mCommandAllocators;
    ComPtr<ID3D12GraphicsCommandList> mCommandList;
    ComPtr<ID3D12Fence> mFence;

    ComPtr<ID3D12RootSignature> mRootSignature;
    ComPtr<ID3D12PipelineState> mPipelineState;

    ComPtr<ID3D12Resource> mTexture;
    ComPtr<ID3D12Resource> mTextureUpload;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT mTextureFootprint{};
    UINT64 mUploadBufferSize{};

    UINT mRtvDescriptorSize{};
    UINT mFrameIndex{};
    std::array<UINT64, FrameCount> mFenceValues{};
    HANDLE mFenceEvent{};

    D3D12_VIEWPORT mViewport{};
    D3D12_RECT mScissorRect{};
};

ATOM D3D12Window::atom = 0;

/** Convert a WinMain command line (lpCmdLine) into a regular argc,argv pair.
 * @param cmdLine Windows API WinMain format command line.
 * @return tuple containing a vector of char* (std::get<0>) and a string
 * containing the argument strings separated each by a null character(std::get<1>).
 */
static std::tuple<std::vector<char*>, std::string> GetArgs(char* cmdLine)
{
    std::tuple<std::vector<char*>, std::string> result;
    std::get<1>(result) = cmdLine ? cmdLine : "";

    if (std::get<1>(result).empty())
    {
        return result;
    }

    for (char& c : std::get<1>(result))
    {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
        {
            c = '\0';
        }
    }

    std::string& storage = std::get<1>(result);
    std::vector<char*>& args = std::get<0>(result);

    bool atStart = true;
    for (size_t i = 0; i < storage.size(); ++i)
    {
        if (storage[i] != '\0' && atStart)
        {
            args.emplace_back(storage.data() + i);
            atStart = false;
        }
        else if (storage[i] == '\0')
        {
            atStart = true;
        }
    }

    return result;
}

int main(int argc, char* argv[])
{
    AeonGUI::Initialize(argc, argv);

    MSG msg{};
    try
    {
        {
            D3D12Window window(
                GetModuleHandle(NULL),
                (argc > 1) ? argv[1] : nullptr,
                800,
                600);

            while (msg.message != WM_QUIT)
            {
                if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
                {
                    if (msg.message != WM_QUIT)
                    {
                        TranslateMessage(&msg);
                        DispatchMessage(&msg);
                    }
                }
                else
                {
                    window.RenderLoop();
                }
            }
        }
    }
    catch (const std::runtime_error& e)
    {
        std::cerr << e.what() << std::endl;
    }

    AeonGUI::Finalize();
    return static_cast<int>(msg.wParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    auto args = GetArgs(lpCmdLine);
    return main(static_cast<int>(std::get<0>(args).size()), std::get<0>(args).data());
}
