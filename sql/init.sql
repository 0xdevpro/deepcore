CREATE TABLE `app` (
  `id` varchar(36) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID',
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Name of the app',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Description of the app',
  `mode` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Mode of the app: function call, ReAct (default)',
  `icon` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Icon URL of the app',
  `status` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Status of the app: draft, active, inactive',
  `role_settings` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Role settings for the agent',
  `welcome_message` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Welcome message for the agent',
  `twitter_link` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Twitter link for the agent',
  `telegram_bot_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Telegram bot ID for the agent',
  `tool_prompt` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Tool prompt for the agent',
  `max_loops` int DEFAULT 3 COMMENT 'Maximum number of loops the agent can perform',
  `is_deleted` tinyint(1) DEFAULT NULL COMMENT 'Logical deletion flag',
  `tenant_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Tenant ID',
  `update_time` datetime DEFAULT (now()) COMMENT 'Last update time',
  `create_time` datetime DEFAULT (now()) COMMENT 'Creation time',
  `model_json` JSON COMMENT 'Additional fields merged into a JSON column',
  `is_public` BOOLEAN DEFAULT FALSE COMMENT 'Whether the agent is public',
  `is_official` BOOLEAN DEFAULT FALSE COMMENT 'Whether the agent is official preset',
  `suggested_questions` JSON COMMENT 'List of suggested questions for the agent',
  `model_id` bigint DEFAULT NULL COMMENT 'ID of the associated model',
  PRIMARY KEY (`id`),
  KEY `idx_tenant` (`tenant_id`),
  KEY `idx_model` (`model_id`),
  KEY `idx_public_official` (`is_public`, `is_official`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


CREATE TABLE `file_storage` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'Auto-incrementing ID',
  `file_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Name of the file',
  `file_uuid` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'file UUID',
  `file_content` blob NOT NULL COMMENT 'Content of the file',
  `size` bigint NOT NULL COMMENT 'Size of the file',
  `create_time` datetime DEFAULT (now()) COMMENT 'Creation time',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


CREATE TABLE `tools` (
  `id` varchar(36) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID',
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Name of the tool',
  `type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Type of the tool: function or openAPI',
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Content of the tool',
  `is_deleted` tinyint(1) DEFAULT NULL COMMENT 'Logical deletion flag',
  `tenant_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Tenant ID',
  `update_time` datetime DEFAULT (now()) COMMENT 'Last update time',
  `create_time` datetime DEFAULT (now()) COMMENT 'Creation time',
  `is_public` BOOLEAN DEFAULT FALSE COMMENT 'Whether the tool is public',
  `is_official` BOOLEAN DEFAULT FALSE COMMENT 'Whether the tool is official preset',
  `auth_config` JSON COMMENT 'Authentication configuration in JSON format',
  PRIMARY KEY (`id`),
  KEY `idx_tenant` (`tenant_id`),
  KEY `idx_public_official` (`is_public`, `is_official`),
  KEY `idx_type` (`type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `users` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'Auto-incrementing ID',
  `username` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Username',
  `email` varchar(120) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Email address',
  `password` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Hashed password',
  `wallet_address` varchar(42) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Ethereum wallet address',
  `nonce` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Nonce for wallet signature',
  `tenant_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Tenant ID',
  `create_time` datetime DEFAULT (now()) COMMENT 'Registration time',
  `update_time` datetime DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update time',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_email` (`email`),
  UNIQUE KEY `uk_wallet_address` (`wallet_address`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `agent_tools` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'Auto-incrementing ID',
  `agent_id` varchar(36) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID of the agent',
  `tool_id` varchar(36) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID of the tool',
  `tenant_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Tenant ID',
  `create_time` datetime DEFAULT (now()) COMMENT 'Creation time',
  PRIMARY KEY (`id`),
  KEY `idx_agent_tool` (`agent_id`, `tool_id`),
  KEY `idx_tenant` (`tenant_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `models` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT 'Auto-incrementing ID',
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Name of the model',
  `endpoint` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'API endpoint of the model',
  `api_key` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'API key for the model',
  `is_official` BOOLEAN DEFAULT FALSE COMMENT 'Whether the model is official preset',
  `is_public` BOOLEAN DEFAULT FALSE COMMENT 'Whether the model is public',
  `tenant_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Tenant ID',
  `create_time` datetime DEFAULT (now()) COMMENT 'Creation time',
  `update_time` datetime DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update time',
  PRIMARY KEY (`id`),
  KEY `idx_tenant` (`tenant_id`),
  KEY `idx_public_official` (`is_public`, `is_official`),
  KEY `idx_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ALTER TABLE `tools`
-- ADD COLUMN `tenant_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Tenant ID' AFTER `is_deleted`;
--
-- ALTER TABLE `users`
-- ADD COLUMN `tenant_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Tenant ID' AFTER `nonce`;
--
-- ALTER TABLE `app`
-- ADD COLUMN `model_json` JSON COMMENT 'Additional fields merged into a JSON column' AFTER `create_time`;
--
-- ALTER TABLE `app` MODIFY COLUMN `id` varchar(36) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID';
--
-- ALTER TABLE `tools` MODIFY COLUMN `app_id` varchar(36) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UUID of the associated app';