//! Message payload types per WIRE_FORMAT_SPEC.md

mod session_init;
mod session_params;
mod base_ot;
mod session_ready;
mod embed_fetch;
mod error_msg;

pub use session_init::OtSessionInitPayload;
pub use session_params::OtSessionParamsPayload;
pub use base_ot::OtBaseOtMsgPayload;
pub use session_ready::OtSessionReadyPayload;
pub use embed_fetch::{EmbedFetchRequestPayload, EmbedFetchResponsePayload};
pub use error_msg::ErrorPayload;

use crate::constants::MsgType;
use crate::error::Result;
use crate::frame::Frame;
use crate::header::Header;

/// Trait for message payloads that can be encoded/decoded
pub trait Payload: Sized {
    /// The message type for this payload
    const MSG_TYPE: MsgType;

    /// Encode the payload to bytes
    fn encode(&self) -> Vec<u8>;

    /// Decode the payload from bytes
    fn decode(data: &[u8]) -> Result<Self>;

    /// Create a frame from this payload and a header
    fn into_frame(self, header: Header) -> Frame {
        Frame::new(Self::MSG_TYPE, header, self.encode())
    }
}

/// A parsed message with its header and typed payload
#[derive(Debug, Clone)]
pub enum Message {
    OtSessionInit {
        header: Header,
        payload: OtSessionInitPayload,
    },
    OtSessionParams {
        header: Header,
        payload: OtSessionParamsPayload,
    },
    OtBaseOtMsg {
        header: Header,
        payload: OtBaseOtMsgPayload,
    },
    OtSessionReady {
        header: Header,
        payload: OtSessionReadyPayload,
    },
    EmbedFetchRequest {
        header: Header,
        payload: EmbedFetchRequestPayload,
    },
    EmbedFetchResponse {
        header: Header,
        payload: EmbedFetchResponsePayload,
    },
    Error {
        header: Header,
        payload: ErrorPayload,
    },
}

impl Message {
    /// Parse a frame into a typed message
    pub fn from_frame(frame: Frame) -> Result<Self> {
        match frame.msg_type {
            MsgType::OtSessionInit => Ok(Message::OtSessionInit {
                header: frame.header,
                payload: OtSessionInitPayload::decode(&frame.payload)?,
            }),
            MsgType::OtSessionParams => Ok(Message::OtSessionParams {
                header: frame.header,
                payload: OtSessionParamsPayload::decode(&frame.payload)?,
            }),
            MsgType::OtBaseOtMsg => Ok(Message::OtBaseOtMsg {
                header: frame.header,
                payload: OtBaseOtMsgPayload::decode(&frame.payload)?,
            }),
            MsgType::OtSessionReady => Ok(Message::OtSessionReady {
                header: frame.header,
                payload: OtSessionReadyPayload::decode(&frame.payload)?,
            }),
            MsgType::EmbedFetchRequest => Ok(Message::EmbedFetchRequest {
                header: frame.header,
                payload: EmbedFetchRequestPayload::decode(&frame.payload)?,
            }),
            MsgType::EmbedFetchResponse => Ok(Message::EmbedFetchResponse {
                header: frame.header,
                payload: EmbedFetchResponsePayload::decode(&frame.payload)?,
            }),
            MsgType::Error => Ok(Message::Error {
                header: frame.header,
                payload: ErrorPayload::decode(&frame.payload)?,
            }),
        }
    }

    /// Encode this message into a frame
    pub fn into_frame(self) -> Frame {
        match self {
            Message::OtSessionInit { header, payload } => payload.into_frame(header),
            Message::OtSessionParams { header, payload } => payload.into_frame(header),
            Message::OtBaseOtMsg { header, payload } => payload.into_frame(header),
            Message::OtSessionReady { header, payload } => payload.into_frame(header),
            Message::EmbedFetchRequest { header, payload } => payload.into_frame(header),
            Message::EmbedFetchResponse { header, payload } => payload.into_frame(header),
            Message::Error { header, payload } => payload.into_frame(header),
        }
    }

    /// Encode this message into bytes
    pub fn encode(&self) -> Vec<u8> {
        self.clone().into_frame().encode()
    }

    /// Decode a message from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        let frame = Frame::decode(data)?;
        Self::from_frame(frame)
    }
}
